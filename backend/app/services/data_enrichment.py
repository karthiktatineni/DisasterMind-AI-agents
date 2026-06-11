from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians
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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result == result else default


def _safe_int(value: Any, default: int = 0) -> int:
    return int(max(0, round(_safe_float(value, float(default)))))


def _latest_world_bank_value(payload: Any) -> float | None:
    if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
        return None
    for row in payload[1]:
        if isinstance(row, dict) and row.get("value") is not None:
            return _safe_float(row["value"])
    return None


def _bbox_area_km2(bounding_box: list[str] | None) -> float:
    if not bounding_box or len(bounding_box) != 4:
        return 0.0
    south, north, west, east = [_safe_float(value) for value in bounding_box]
    height = abs(north - south) * 111.32
    mean_lat = radians((north + south) / 2)
    width = abs(east - west) * 111.32 * max(0.1, cos(mean_lat))
    return max(0.0, height * width)


def _tag_capacity(tags: dict[str, Any]) -> int:
    for key in ("capacity:beds", "beds", "capacity"):
        value = tags.get(key)
        if value is None:
            continue
        text = str(value).replace(",", "").strip()
        if text.isdigit():
            return int(text)
    return 0


async def enrich_scenario(intake: ScenarioIntakeRequest) -> DisasterScenario:
    sources: list[dict[str, str]] = []
    latitude: float | None = None
    longitude: float | None = None
    country: str | None = None
    resolved_location: str | None = None
    population: int | None = None
    population_area_km2 = 0.0
    country_code: str | None = None
    country_iso3: str | None = None
    country_population: int | None = None
    country_area_km2 = 0.0
    beds_per_1000: float | None = None
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
            country_code = result.get("country_code")
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
                nominatim = await client.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={
                        "q": intake.region,
                        "format": "jsonv2",
                        "limit": 1,
                        "addressdetails": 1,
                        "extratags": 1,
                    },
                )
                nominatim.raise_for_status()
                place = (nominatim.json() or [])[0]
                tags = place.get("extratags", {})
                osm_population = _safe_int(tags.get("population"))
                if osm_population and not population:
                    population = osm_population
                population_area_km2 = _bbox_area_km2(place.get("boundingbox"))
                sources.append(_source("Nominatim", "live", "Fetched OSM place metadata and bounding box."))
            except Exception:
                sources.append(_source("Nominatim", "fallback", "OSM place metadata unavailable."))

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
                hospital_facilities = 0
                shelter_facilities = 0
                for element in elements:
                    tags = element.get("tags", {})
                    cap = _tag_capacity(tags)

                    if tags.get("amenity") in ("hospital", "clinic"):
                        hospital_facilities += 1
                        if not cap:
                            cap = 150 if tags.get("amenity") == "hospital" else 25
                        hospital_capacity += cap
                    if tags.get("social_facility") == "shelter" or tags.get("emergency") == "assembly_point":
                        shelter_facilities += 1
                        if not cap:
                            cap = 300 if tags.get("social_facility") == "shelter" else 500
                        shelter_capacity += cap
                sources.append(
                    _source(
                        "OpenStreetMap Overpass",
                        "live",
                        (
                            f"Fetched {hospital_facilities} medical facilities and "
                            f"{shelter_facilities} shelter or assembly sites nearby."
                        ),
                    )
                )
            except Exception:
                sources.append(_source("OpenStreetMap Overpass", "fallback", "Facility lookup unavailable."))

        if country_code:
            try:
                country_response = await client.get(
                    f"https://restcountries.com/v3.1/alpha/{country_code}",
                    params={"fields": "population,area,cca3,name"},
                )
                country_response.raise_for_status()
                country_data = country_response.json()
                if isinstance(country_data, list):
                    country_data = country_data[0]
                country_population = _safe_int(country_data.get("population"))
                country_area_km2 = _safe_float(country_data.get("area"))
                country_iso3 = country_data.get("cca3")
                sources.append(_source("REST Countries", "live", "Fetched country population and area."))
            except Exception:
                sources.append(_source("REST Countries", "fallback", "Country population API unavailable."))

        if country_iso3:
            try:
                beds_response = await client.get(
                    f"https://api.worldbank.org/v2/country/{country_iso3}/indicator/SH.MED.BEDS.ZS",
                    params={"format": "json", "per_page": 10},
                )
                beds_response.raise_for_status()
                beds_per_1000 = _latest_world_bank_value(beds_response.json())
                if beds_per_1000 is not None:
                    sources.append(_source("World Bank", "latest", "Fetched latest hospital beds per 1,000 people."))
            except Exception:
                sources.append(_source("World Bank", "fallback", "Hospital bed density API unavailable."))

    population = population or country_population or _fallback_population(intake.region, intake.severity)
    if not any(source["name"] in {"Open-Meteo Geocoding", "Nominatim", "REST Countries"} and source["status"] in {"live", "latest"} for source in sources):
        sources.append(_source("Population estimate", "fallback", "Estimated population from scenario region text."))

    area_km2 = population_area_km2 or country_area_km2
    density = round(population / area_km2, 2) if area_km2 else round(population / 300, 2)
    if area_km2:
        sources.append(_source("Population density", "derived", "Computed from live population and area APIs."))
    else:
        sources.append(_source("Population density", "derived", "Estimated because no live area was available."))

    if hospital_capacity == 0 and population:
        bed_ratio = beds_per_1000 if beds_per_1000 is not None else 2.5
        hospital_capacity = max(1, round(population * bed_ratio / 1000))
        sources.append(_source("Hospital capacity", "derived", "Estimated from latest country hospital-bed density."))

    if shelter_capacity == 0 and population:
        shelter_fraction = {
            "Low": 0.04,
            "Moderate": 0.08,
            "High": 0.14,
            "Critical": 0.2,
        }[intake.severity]
        shelter_capacity = max(1, round(population * shelter_fraction))
        sources.append(_source("Shelter capacity", "derived", "Estimated from population and scenario severity."))

    base_damage = {"Low": 0.2, "Moderate": 0.4, "High": 0.7, "Critical": 0.95}
    historical_damage = base_damage.get(intake.severity, 0.5)

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
