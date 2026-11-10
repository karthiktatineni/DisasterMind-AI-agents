"""Concrete real-work domain tools for Search, Historical Stats, Open-Meteo, and verified WHO/Sphere water standard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
from pydantic import Field

from backend.app.services.historical_stats import HistoricalStatsService
from backend.app.services.retrieval_service import RetrievalService
from backend.app.tools.registry import StrictArgsBase, ToolSpec

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STANDARDS_PATH = PROJECT_ROOT / "config" / "planning_standards.json"


class SearchPrecedentsArgs(StrictArgsBase):
    query: str = Field(min_length=3, max_length=500)
    disaster_type: str | None = None
    country: str | None = None
    year_min: int | None = Field(default=None, ge=1900, le=2030)
    year_max: int | None = Field(default=None, ge=1900, le=2030)
    top_k: int = Field(default=5, ge=1, le=10)


class GetHistoricalStatsArgs(StrictArgsBase):
    disaster_type: str | None = None
    country: str | None = None
    year_min: int | None = Field(default=None, ge=1900, le=2030)
    year_max: int | None = Field(default=None, ge=1900, le=2030)


class GeocodeRegionArgs(StrictArgsBase):
    region: str = Field(min_length=2, max_length=200)


class GetWeatherArgs(StrictArgsBase):
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)


class ComputeLogisticsArgs(StrictArgsBase):
    population: int = Field(gt=0)
    population_provenance: str = Field(min_length=3)
    available_shelter_capacity: int | None = Field(default=None, ge=0)
    available_water_litres_per_day: float | None = Field(default=None, ge=0.0)


class DelegateFocusArgs(StrictArgsBase):
    focus_question: str = Field(min_length=3, max_length=500)


def load_planning_standards(path: Path | None = None) -> dict[str, Any]:
    target = path or STANDARDS_PATH
    return json.loads(target.read_text(encoding="utf-8"))


def parse_region_query(region: str) -> tuple[str, str | None]:
    """Parse optional 'place, country' format into (place_name, country_filter)."""
    cleaned = " ".join(region.strip().split())
    if "," not in cleaned:
        return cleaned, None
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    if len(parts) < 2:
        return parts[0] if parts else cleaned, None
    return parts[0], parts[-1]


def _pick_country_match(results: list[dict[str, Any]], country_filter: str) -> dict[str, Any] | None:
    needle = country_filter.strip().lower()
    for row in results:
        c_name = str(row.get("country") or "").strip().lower()
        c_code = str(row.get("country_code") or "").strip().lower()
        if c_name == needle or needle in c_name or c_code == needle:
            return row
    return None


def build_domain_tools(
    *,
    retrieval_service: RetrievalService,
    historical_stats_service: HistoricalStatsService | None,
    http_client: httpx.AsyncClient,
    standards_path: Path | None = None,
) -> list[ToolSpec]:
    """Create validated real-work domain tool specifications (excluding delegation hooks)."""
    standards_doc = load_planning_standards(standards_path)
    standards = standards_doc["standards"]

    async def handle_search_precedents(args: dict[str, Any], call_id: str) -> tuple[str, dict[str, Any], str | None]:
        res = await retrieval_service.search(
            args["query"],
            top_k=args["top_k"],
            disaster_type=args.get("disaster_type"),
            country=args.get("country"),
            year_min=args.get("year_min"),
            year_max=args.get("year_max"),
        )
        if res.status == "error":
            return "error", {"threshold_used": res.threshold_used, "namespace": res.namespace}, res.reason
        if res.status == "no_data":
            return (
                "no_data",
                {
                    "call_id": call_id,
                    "matches": [],
                    "record_ids": [],
                    "threshold_used": res.threshold_used,
                    "top_score": res.top_score,
                    "dataset_version": res.dataset_version,
                    "reason": res.reason or "No precedent matched above calibrated threshold.",
                },
                None,
            )
        # Allow-list structured fields only — raw search_text is NEVER exposed to LLMs (Task 6).
        safe_matches = [
            {
                "record_id": m.id,
                "event_name": m.event_name,
                "disaster_type": m.disaster_type,
                "country": m.country,
                "region": m.region,
                "start_year": m.start_year,
                "total_deaths": m.total_deaths,
                "total_affected": m.total_affected,
                "total_damage_usd_thousands": m.total_damage,
                "similarity": m.similarity,
            }
            for m in res.matches
        ]
        return (
            "ok",
            {
                "call_id": call_id,
                "matches": safe_matches,
                "record_ids": [m["record_id"] for m in safe_matches],
                "threshold_used": res.threshold_used,
                "top_score": res.top_score,
                "dataset_version": res.dataset_version,
            },
            None,
        )

    async def handle_get_historical_stats(args: dict[str, Any], call_id: str) -> tuple[str, dict[str, Any], str | None]:
        if historical_stats_service is None:
            return "error", {}, "HistoricalStatsService unavailable (canonical snapshot not loaded)."
        stats = historical_stats_service.query_stats(
            disaster_type=args.get("disaster_type"),
            country=args.get("country"),
            start_year=args.get("year_min"),
            end_year=args.get("year_max"),
        )
        stats["call_id"] = call_id
        status = "ok" if stats.get("status") == "ok" else "no_data"
        return status, stats, None

    async def _query_open_meteo_geo(name: str, count: int) -> list[dict[str, Any]]:
        resp = await http_client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": name, "count": count, "language": "en", "format": "json"},
        )
        resp.raise_for_status()
        return list(resp.json().get("results") or [])

    async def handle_geocode_region(args: dict[str, Any], call_id: str) -> tuple[str, dict[str, Any], str | None]:
        raw_region = args["region"].strip()
        place, country_filter = parse_region_query(raw_region)
        results = await _query_open_meteo_geo(raw_region, 5)
        top = _pick_country_match(results, country_filter) if (results and country_filter) else (results[0] if results else None)

        if top is None and country_filter:
            fallback_results = await _query_open_meteo_geo(place, 10)
            top = _pick_country_match(fallback_results, country_filter)

        if top is None:
            return "no_data", {"call_id": call_id, "region": raw_region}, f"No geocoding match for '{raw_region}'."

        resolved = ", ".join(p for p in [top.get("name"), top.get("admin1"), top.get("country")] if p)
        pop_val = top.get("population")
        return (
            "ok",
            {
                "call_id": call_id,
                "latitude": float(top["latitude"]),
                "longitude": float(top["longitude"]),
                "country": top.get("country"),
                "resolved_location": resolved,
                "population": int(pop_val) if pop_val is not None else None,
                "source": "Open-Meteo Geocoding API",
            },
            None,
        )

    async def handle_get_weather(args: dict[str, Any], call_id: str) -> tuple[str, dict[str, Any], str | None]:
        resp = await http_client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": args["latitude"],
                "longitude": args["longitude"],
                "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
                "wind_speed_unit": "kmh",
                "timezone": "UTC",
            },
        )
        resp.raise_for_status()
        current = resp.json().get("current") or {}
        if not current:
            return "no_data", {"call_id": call_id}, "Open-Meteo forecast returned empty current conditions."
        return (
            "ok",
            {
                "call_id": call_id,
                "latitude": args["latitude"],
                "longitude": args["longitude"],
                "temperature_c": current.get("temperature_2m"),
                "relative_humidity_pct": current.get("relative_humidity_2m"),
                "precipitation_mm": current.get("precipitation"),
                "wind_speed_kmh": current.get("wind_speed_10m"),
                "source": "Open-Meteo Forecast API",
            },
            None,
        )

    async def handle_compute_logistics(args: dict[str, Any], call_id: str) -> tuple[str, dict[str, Any], str | None]:
        pop = int(args["population"])
        pop_prov = str(args["population_provenance"]).strip()
        water_std_meta = standards["water_litres_per_person_per_day"]
        water_std = float(water_std_meta["value"])
        water_req = round(pop * water_std, 2)

        quantities = [
            {
                "name": "daily_minimum_water_requirement",
                "value": water_req,
                "unit": "litres/day",
                "provenance": call_id,
                "inputs": [
                    f"population={pop} ({pop_prov})",
                    f"standard:water_litres_per_person_per_day={water_std} ({standards_doc['url']}, retrieved {standards_doc['retrieved_date']})",
                ],
            }
        ]

        avail_water = args.get("available_water_litres_per_day")
        avail_shelter = args.get("available_shelter_capacity")

        comparisons = [
            {
                "resource": "daily_potable_water",
                "required": water_req,
                "required_unit": "litres/day",
                "available": avail_water,
                "available_status": "provided_by_user" if avail_water is not None else "not_provided",
                "deficit_or_surplus": round(avail_water - water_req, 2) if avail_water is not None else None,
                "provenance": call_id,
                "inputs": [f"population={pop} ({pop_prov})", "standard:water_litres_per_person_per_day=15.0"],
            },
            {
                "resource": "sheltered_population_capacity",
                "required": pop,
                "required_unit": "people (1:1 exposed population headcount)",
                "available": avail_shelter,
                "available_status": "provided_by_user" if avail_shelter is not None else "not_provided",
                "deficit_or_surplus": (avail_shelter - pop) if avail_shelter is not None else None,
                "provenance": call_id,
                "inputs": [f"population={pop} ({pop_prov})"],
            },
        ]
        return (
            "ok",
            {
                "call_id": call_id,
                "standards_citation": f"{standards_doc['source']} ({standards_doc['edition']}), {standards_doc['url']}, retrieved {standards_doc['retrieved_date']}",
                "quantities": quantities,
                "capacity_comparisons": comparisons,
            },
            None,
        )

    return [
        ToolSpec(
            name="search_precedents",
            description="Search calibrated EM-DAT historical precedents in Pinecone. Returns allow-listed structured metadata and record_ids.",
            args_model=SearchPrecedentsArgs,
            handler=handle_search_precedents,
            timeout_s=15.0,
        ),
        ToolSpec(
            name="get_historical_stats",
            description="Compute EM-DAT snapshot historical medians and p90 impact metrics (enforces n>=5 no_data guardrail).",
            args_model=GetHistoricalStatsArgs,
            handler=handle_get_historical_stats,
            timeout_s=10.0,
        ),
        ToolSpec(
            name="geocode_region",
            description="Resolve a region name (supports 'place, country') to coordinates and optional population via Open-Meteo Geocoding.",
            args_model=GeocodeRegionArgs,
            handler=handle_geocode_region,
            timeout_s=10.0,
        ),
        ToolSpec(
            name="get_weather",
            description="Fetch live current meteorological conditions from Open-Meteo for latitude and longitude.",
            args_model=GetWeatherArgs,
            handler=handle_get_weather,
            timeout_s=10.0,
        ),
        ToolSpec(
            name="compute_logistics_requirements",
            description="Derive verified WHO/Sphere daily water requirement (15 L/person/day) from verified population and compare optional user inventory.",
            args_model=ComputeLogisticsArgs,
            handler=handle_compute_logistics,
            timeout_s=5.0,
        ),
    ]
