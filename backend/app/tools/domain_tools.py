"""Concrete real-work domain tools for Search, Historical Stats, Open-Meteo, and Sphere Logistics."""

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
    available_food_kcal_per_day: float | None = Field(default=None, ge=0.0)


class DelegateFocusArgs(StrictArgsBase):
    focus_question: str = Field(min_length=3, max_length=500)


def load_planning_standards(path: Path | None = None) -> dict[str, Any]:
    target = path or STANDARDS_PATH
    return json.loads(target.read_text(encoding="utf-8"))


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

    async def handle_geocode_region(args: dict[str, Any], call_id: str) -> tuple[str, dict[str, Any], str | None]:
        resp = await http_client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": args["region"], "count": 1, "language": "en", "format": "json"},
        )
        resp.raise_for_status()
        payload = resp.json()
        results = payload.get("results") or []
        if not results:
            return "no_data", {"call_id": call_id, "region": args["region"]}, f"No geocoding match for '{args['region']}'."
        top = results[0]
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
        water_std = float(standards["water_litres_per_person_per_day"]["value"])
        shelter_std = float(standards["shelter_covered_area_m2_per_person"]["value"])
        food_std = float(standards["food_kcal_per_person_per_day"]["value"])

        water_req = round(pop * water_std, 2)
        shelter_area_req = round(pop * shelter_std, 2)
        food_req = round(pop * food_std, 2)

        quantities = [
            {
                "name": "daily_minimum_water_requirement",
                "value": water_req,
                "unit": "litres/day",
                "provenance": f"{call_id}",
                "inputs": [f"population={pop} ({pop_prov})", "standard:sphere_2018_wash_15L (15.0 L/person/day)"],
            },
            {
                "name": "minimum_shelter_covered_area",
                "value": shelter_area_req,
                "unit": "m2",
                "provenance": f"{call_id}",
                "inputs": [f"population={pop} ({pop_prov})", "standard:sphere_2018_shelter_3_5m2 (3.5 m2/person)"],
            },
            {
                "name": "daily_food_energy_requirement",
                "value": food_req,
                "unit": "kcal/day",
                "provenance": f"{call_id}",
                "inputs": [f"population={pop} ({pop_prov})", "standard:sphere_2018_food_2100kcal (2100 kcal/person/day)"],
            },
        ]

        avail_shelter = args.get("available_shelter_capacity")
        avail_water = args.get("available_water_litres_per_day")
        avail_food = args.get("available_food_kcal_per_day")

        comparisons = [
            {
                "resource": "sheltered_population_capacity",
                "required": pop,
                "required_unit": "people (at 3.5 m2/person covered area)",
                "available": avail_shelter,
                "available_status": "provided_by_user" if avail_shelter is not None else "not_provided",
                "deficit_or_surplus": (avail_shelter - pop) if avail_shelter is not None else None,
                "provenance": call_id,
                "inputs": [f"population={pop} ({pop_prov})", "standard:sphere_2018_shelter_3_5m2"],
            },
            {
                "resource": "daily_potable_water",
                "required": water_req,
                "required_unit": "litres/day",
                "available": avail_water,
                "available_status": "provided_by_user" if avail_water is not None else "not_provided",
                "deficit_or_surplus": round(avail_water - water_req, 2) if avail_water is not None else None,
                "provenance": call_id,
                "inputs": [f"population={pop} ({pop_prov})", "standard:sphere_2018_wash_15L"],
            },
            {
                "resource": "daily_food_energy",
                "required": food_req,
                "required_unit": "kcal/day",
                "available": avail_food,
                "available_status": "provided_by_user" if avail_food is not None else "not_provided",
                "deficit_or_surplus": round(avail_food - food_req, 2) if avail_food is not None else None,
                "provenance": call_id,
                "inputs": [f"population={pop} ({pop_prov})", "standard:sphere_2018_food_2100kcal"],
            },
        ]
        return (
            "ok",
            {
                "call_id": call_id,
                "standards_citation": f"{standards_doc['source']} ({standards_doc['edition']}), retrieved {standards_doc['retrieved_date']}",
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
            description="Resolve a region name to live coordinates, country, and optional census population via Open-Meteo Geocoding.",
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
            description="Derive Sphere 2018 minimum water/shelter/food requirements from verified population and compare optional user inventory.",
            args_model=ComputeLogisticsArgs,
            handler=handle_compute_logistics,
            timeout_s=5.0,
        ),
    ]
