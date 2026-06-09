from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from backend.app.schemas import DisasterScenario, ScenarioIntakeRequest


SEVERITY_SCALE = {
    "Low": 0.45,
    "Moderate": 0.7,
    "High": 1.0,
    "Critical": 1.35,
}


@dataclass(frozen=True)
class EnrichmentResult:
    scenario: DisasterScenario


def _source(name: str, status: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def _fallback_population(region: str, severity: str) -> int:
    return round((65000 + max(1, len(region)) * 2200) * SEVERITY_SCALE[severity])


async def enrich_scenario(intake: ScenarioIntakeRequest) -> DisasterScenario:
    sources: list[dict[str, str]] = []
    latitude: float | None = None
    longitude: float | None = None
    country: str | None = None
    resolved_location: str | None = None
    population: int | None = None
    temperature = 0.0
    humidity = 0.0
    wind_speed = 0.0
    rainfall = 0.0
    elevation = 0.0
    hospital_capacity = 0
    shelter_capacity = 0

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(10.0),
        headers={"User-Agent": "DisasterMindAI/0.1 data-enrichment"},
    ) as client:
        try:
            geo = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": intake.region, "count": 1, "language": "en", "format": "json"},
            )
            geo.raise_for_status()
            result = (geo.json().get("results") or [])[0]
            latitude = float(result["latitude"])
            longitude = float(result["longitude"])
            country = result.get("country")
            resolved_location = ", ".join(
                part for part in [result.get("name"), result.get("admin1"), country] if part
            )
            population = result.get("population")
            population = int(population) if population else None
            sources.append(_source("Open-Meteo Geocoding", "live", "Resolved region globally."))
        except Exception:
            sources.append(_source("Open-Meteo Geocoding", "fallback", "Could not resolve region live."))

        if latitude is not None and longitude is not None:
            try:
                forecast = await client.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": latitude,
                        "longitude": longitude,
                        "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
                        "wind_speed_unit": "kmh",
                        "timezone": "auto",
                    },
                )
                forecast.raise_for_status()
                current = forecast.json().get("current", {})
                temperature = float(current.get("temperature_2m") or temperature)
                humidity = float(current.get("relative_humidity_2m") or humidity)
                wind_speed = float(current.get("wind_speed_10m") or wind_speed)
                rainfall = float(current.get("precipitation") or rainfall)
                sources.append(_source("Open-Meteo Forecast", "live", "Fetched current weather globally."))
            except Exception:
                sources.append(_source("Open-Meteo Forecast", "fallback", "Weather service unavailable."))

            try:
                elev = await client.get(
                    "https://api.open-meteo.com/v1/elevation",
                    params={"latitude": latitude, "longitude": longitude},
                )
                elev.raise_for_status()
                values = elev.json().get("elevation") or []
                elevation = float(values[0]) if values else elevation
                sources.append(_source("Open-Meteo Elevation", "live", "Fetched 90m DEM elevation."))
            except Exception:
                sources.append(_source("Open-Meteo Elevation", "fallback", "Elevation service unavailable."))

            try:
                overpass_query = f"""
                [out:json][timeout:8];
                (
                  node(around:25000,{latitude},{longitude})[amenity=hospital];
                  way(around:25000,{latitude},{longitude})[amenity=hospital];
                  relation(around:25000,{latitude},{longitude})[amenity=hospital];
                  node(around:25000,{latitude},{longitude})[amenity=clinic];
                  way(around:25000,{latitude},{longitude})[amenity=clinic];
                  node(around:25000,{latitude},{longitude})[social_facility=shelter];
                  way(around:25000,{latitude},{longitude})[social_facility=shelter];
                  node(around:25000,{latitude},{longitude})[emergency=assembly_point];
                  way(around:25000,{latitude},{longitude})[emergency=assembly_point];
                );
                out tags;
                """
                osm = await client.post(
                    "https://overpass-api.de/api/interpreter",
                    data={"data": overpass_query},
                )
                osm.raise_for_status()
                elements: list[dict[str, Any]] = osm.json().get("elements", [])
                for element in elements:
                    tags = element.get("tags", {})
                    
                    # Attempt to get real capacity if listed, otherwise default 0
                    try:
                        cap = int(tags.get("capacity", 0))
                    except ValueError:
                        cap = 0

                    if tags.get("amenity") in ("hospital", "clinic"):
                        hospital_capacity += cap
                    if tags.get("social_facility") == "shelter" or tags.get("emergency") == "assembly_point":
                        shelter_capacity += cap
                sources.append(_source("OpenStreetMap Overpass", "live", "Fetched nearby hospitals, clinics, and shelters."))
            except Exception:
                sources.append(_source("OpenStreetMap Overpass", "fallback", "Facility lookup unavailable."))

    severity_scale = SEVERITY_SCALE[intake.severity]
    disaster = intake.disaster_type.lower()
    population = population or 0

    density = 0.0
    historical_damage = 0.0

    scenario = DisasterScenario(
        disaster_type=intake.disaster_type,
        region=intake.region,
        time_horizon=intake.time_horizon,
        severity=intake.severity,
        population=population,
        hospital_capacity=hospital_capacity,
        shelter_capacity=shelter_capacity,
        wind_speed=wind_speed,
        rainfall=rainfall,
        elevation=elevation,
        population_density=density,
        temperature=temperature,
        humidity=humidity,
        historical_damage=historical_damage,
        notes=intake.notes,
        latitude=latitude,
        longitude=longitude,
        country=country,
        resolved_location=resolved_location,
        data_sources=sources,
    )
    return scenario
