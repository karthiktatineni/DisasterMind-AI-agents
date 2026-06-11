from __future__ import annotations

import json
import math
from typing import Any

from backend.app.schemas import DisasterScenario


SEVERITY_BASE = {
    "Low": 20,
    "Moderate": 45,
    "High": 70,
    "Critical": 90,
}


def parse_json_object(content: str) -> dict[str, Any]:
    """Parse strict JSON, with a small recovery path for fenced model output."""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(content[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Model response must be a JSON object.")
    return parsed


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def safe_int(value: Any, default: int = 0) -> int:
    return int(max(0, round(safe_float(value, float(default)))))


def clamp(value: float, low: int = 0, high: int = 100) -> int:
    return int(max(low, min(high, round(value))))


def retrieved_disasters(rag: dict[str, Any]) -> list[dict[str, Any]]:
    rows = rag.get("retrieved_disasters", [])
    return [row for row in rows if isinstance(row, dict)]


def top_similarity(rag: dict[str, Any]) -> float:
    rows = retrieved_disasters(rag)
    return safe_float(rows[0].get("similarity")) if rows else 0.0


def evidence_citations(rag: dict[str, Any], limit: int = 5) -> list[str]:
    citations = []
    for row in retrieved_disasters(rag)[:limit]:
        name = row.get("event_name") or row.get("id") or "Retrieved disaster"
        year = row.get("start_year") or "unknown year"
        similarity = round(safe_float(row.get("similarity")) * 100)
        citations.append(f"{name} ({year}) similarity {similarity}%")
    return citations


def weighted_mean(rows: list[dict[str, Any]], field: str) -> float:
    weighted_total = 0.0
    total_weight = 0.0
    for row in rows:
        value = safe_float(row.get(field), -1.0)
        if value < 0:
            continue
        weight = max(0.01, safe_float(row.get("similarity"), 0.01))
        weighted_total += value * weight
        total_weight += weight
    return weighted_total / total_weight if total_weight else 0.0


def risk_level(risk_score: int) -> str:
    if risk_score >= 85:
        return "critical"
    if risk_score >= 65:
        return "high"
    if risk_score >= 35:
        return "medium"
    return "low"


def scenario_place(scenario: DisasterScenario) -> str:
    return scenario.resolved_location or scenario.region or "affected area"


def predicted_affected(prediction: dict[str, Any], scenario: DisasterScenario) -> int:
    affected = safe_int(
        prediction.get("predicted_affected", prediction.get("evacuation_need")),
        0,
    )
    if affected:
        return affected
    if scenario.population:
        severity_fraction = {
            "Low": 0.1,
            "Moderate": 0.25,
            "High": 0.5,
            "Critical": 0.75,
        }.get(scenario.severity, 0.25)
        return safe_int(scenario.population * severity_fraction)
    return 0


def predicted_deaths(prediction: dict[str, Any]) -> int:
    return safe_int(prediction.get("predicted_deaths", prediction.get("casualty_estimate")), 0)


def build_risk_profile(scenario: DisasterScenario, rag: dict[str, Any]) -> dict[str, Any]:
    rows = retrieved_disasters(rag)
    severity_score = SEVERITY_BASE.get(scenario.severity, 45)
    historical_score = top_similarity(rag) * 20
    weather_score = min(15.0, safe_float(scenario.wind_speed) / 18 + safe_float(scenario.rainfall) / 80)
    density_score = min(10.0, safe_float(scenario.population_density) / 1000)
    capacity_score = 0.0
    if scenario.population and scenario.shelter_capacity < scenario.population:
        capacity_score = min(10.0, ((scenario.population - scenario.shelter_capacity) / scenario.population) * 10)

    risk_score = clamp(severity_score + historical_score + weather_score + density_score + capacity_score - 10)
    historical_affected = safe_int(weighted_mean(rows, "total_affected"))
    if scenario.population:
        affected_estimate = min(
            scenario.population,
            max(safe_int(scenario.population * risk_score / 100), safe_int(historical_affected * 0.02)),
        )
    else:
        affected_estimate = historical_affected

    place = scenario_place(scenario)
    contributors = {
        "severity": clamp(severity_score),
        "historical_similarity": clamp(historical_score),
        "weather_exposure": clamp(weather_score),
        "population_density": clamp(density_score),
        "capacity_gap": clamp(capacity_score),
    }

    return {
        "risk_score": risk_score,
        "risk_level": risk_level(risk_score),
        "contributors": contributors,
        "affected_population_estimate": affected_estimate,
        "affected_population": affected_estimate,
        "critical_zones": [
            {
                "zone": place,
                "reason": f"{scenario.severity} {scenario.disaster_type} with {len(rows)} retrieved historical matches.",
                "priority": 1,
            }
        ],
        "generation_mode": "deterministic_fallback",
    }


def build_explanations(
    prediction: dict[str, Any],
    feature_importance: dict[str, Any],
    rag: dict[str, Any],
    risk: dict[str, Any],
) -> dict[str, Any]:
    numeric_importance = [
        (name, safe_float(score))
        for name, score in feature_importance.items()
        if isinstance(score, (int, float))
    ]
    numeric_importance.sort(key=lambda item: item[1], reverse=True)
    feature_reasons = [
        f"{name.replace('_', ' ')} is a top model feature."
        for name, _ in numeric_importance[:3]
    ]
    if not feature_reasons:
        feature_reasons = ["Scenario exposure, capacity, and retrieved historical matches drive the estimate."]

    citations = evidence_citations(rag, limit=3)
    citation_reason = f"Nearest evidence: {', '.join(citations)}." if citations else "No citation rows available."
    risk_reason = f"Risk score {risk.get('risk_score', 0)} ({risk.get('risk_level', 'unknown')})."

    explanations = {
        "deaths": {
            "prediction": predicted_deaths(prediction),
            "because": [risk_reason, *feature_reasons, citation_reason],
        },
        "affected": {
            "prediction": safe_int(prediction.get("predicted_affected")),
            "because": [risk_reason, *feature_reasons, citation_reason],
        },
        "damage": {
            "prediction": safe_int(prediction.get("predicted_damage")),
            "because": [risk_reason, *feature_reasons, citation_reason],
        },
    }
    return {
        "explanations": explanations,
        "feature_importance": feature_importance,
        "generation_mode": "deterministic_fallback",
    }


def build_logistics_plan(
    scenario: DisasterScenario,
    prediction: dict[str, Any],
    risk: dict[str, Any],
) -> dict[str, Any]:
    affected = predicted_affected(prediction, scenario)
    deaths = predicted_deaths(prediction)
    resource_plan = {
        "ambulances": max(1, math.ceil(max(deaths, affected * 0.015) / 20)) if affected or deaths else 0,
        "rescue_teams": max(1, math.ceil(max(affected, 1) / 5000)) if affected else 0,
        "medical_kits": max(1, math.ceil(max(affected * 0.1, deaths * 3) / 100)) if affected or deaths else 0,
        "water_units": safe_int(affected * 3),
        "food_packets": safe_int(affected * 2),
    }

    shortfalls: list[str] = []
    if affected > scenario.shelter_capacity:
        shortfalls.append(f"Shelter capacity short by {affected - scenario.shelter_capacity}.")
    if deaths > scenario.hospital_capacity:
        shortfalls.append("Hospital capacity is below predicted casualties.")
    if risk.get("risk_level") in {"high", "critical"}:
        shortfalls.append("High-risk profile requires pre-positioned rescue teams.")

    return {
        "resource_plan": resource_plan,
        "available_resources": resource_plan,
        "resource_shortfalls": shortfalls,
        "source_scenario": scenario.model_dump(),
        "risk_score": risk.get("risk_score"),
        "generation_mode": "deterministic_fallback",
    }


def build_evacuation_strategy(
    scenario: DisasterScenario,
    prediction: dict[str, Any],
    risk: dict[str, Any],
    rag: dict[str, Any],
) -> dict[str, Any]:
    affected = predicted_affected(prediction, scenario)
    place = scenario_place(scenario)
    phases = [
        ("Priority medical and high-exposure households", 1, 0.4),
        ("Dense residential and low-mobility areas", 2, 0.35),
        ("Remaining exposed population", 3, 0.25),
    ]
    evacuation_zones = [
        {"zone": f"{place} - {name}", "phase": phase, "population": safe_int(affected * share)}
        for name, phase, share in phases
        if affected
    ]
    route_conflicts = []
    if scenario.notes:
        route_conflicts.append(f"Operational note requires routing review: {scenario.notes[:160]}")
    if risk.get("risk_level") in {"high", "critical"}:
        route_conflicts.append("Prioritize redundant routes for high-risk zones.")

    return {
        "evacuation_zones": evacuation_zones,
        "shelter_gap": max(0, affected - scenario.shelter_capacity),
        "route_conflicts": route_conflicts,
        "historical_precedents": evidence_citations(rag, limit=2),
        "risk_level": risk.get("risk_level"),
        "generation_mode": "deterministic_fallback",
    }


def build_simulation_results(
    scenario: DisasterScenario,
    evacuation: dict[str, Any],
    logistics: dict[str, Any],
) -> dict[str, Any]:
    shelter_gap = safe_int(evacuation.get("shelter_gap"))
    resource_shortfalls = logistics.get("resource_shortfalls", [])
    resource_shortfalls = resource_shortfalls if isinstance(resource_shortfalls, list) else []
    water_units = safe_int(logistics.get("resource_plan", {}).get("water_units"))
    affected = sum(safe_int(zone.get("population")) for zone in evacuation.get("evacuation_zones", []))

    scenario_results = [
        {
            "scenario": "Shelter saturation",
            "status": "warning" if shelter_gap else "passed",
            "delta": f"{shelter_gap} people over current shelter capacity." if shelter_gap else "Shelter capacity covers planned evacuation.",
        },
        {
            "scenario": "Medical surge",
            "status": "warning" if resource_shortfalls else "passed",
            "delta": "; ".join(map(str, resource_shortfalls)) if resource_shortfalls else "No resource shortfall flagged.",
        },
        {
            "scenario": "Supply continuity",
            "status": "warning" if affected and water_units < affected * 2 else "passed",
            "delta": "Increase water staging for a 24-hour buffer." if affected and water_units < affected * 2 else "Water staging meets initial response target.",
        },
    ]
    return {
        "scenario_results": scenario_results,
        "resilience_gaps": [row for row in scenario_results if row["status"] == "warning"],
        "logistics_plan": logistics.get("resource_plan", {}),
        "generation_mode": "deterministic_fallback",
    }


def build_planning_output(
    scenario: DisasterScenario,
    risk: dict[str, Any],
    prediction: dict[str, Any],
    logistics: dict[str, Any],
    evacuation: dict[str, Any],
    rag: dict[str, Any],
) -> dict[str, Any]:
    affected = predicted_affected(prediction, scenario)
    primary = min(scenario.shelter_capacity, affected)
    overflow = max(0, affected - primary)
    citations = evidence_citations(rag, limit=3)
    resource_plan = logistics.get("resource_plan", {})
    return {
        "evacuation_strategy": (
            f"Run a phased evacuation for {scenario_place(scenario)} with priority zones first "
            f"and overflow shelter activation if needed."
        ),
        "shelter_assignments": {"primary": primary, "overflow": overflow},
        "resource_estimates": {
            "ambulances": safe_int(resource_plan.get("ambulances")),
            "water_units": safe_int(resource_plan.get("water_units")),
            "food_packets": safe_int(resource_plan.get("food_packets")),
        },
        "phases": [
            {
                "phase": "0-6 hours",
                "action": "Open command post, verify shelters, and stage medical transport.",
                "evidence_citations": citations[:2],
            },
            {
                "phase": "6-18 hours",
                "action": "Move priority evacuation zones and monitor shelter saturation.",
                "evidence_citations": citations[:3],
            },
            {
                "phase": scenario.time_horizon,
                "action": "Reassess risk, resupply critical resources, and update public guidance.",
                "evidence_citations": citations[:3],
            },
        ],
        "risk_level": risk.get("risk_level"),
        "evacuation": evacuation,
        "generation_mode": "deterministic_fallback",
    }


def build_decision_output(
    scenario: DisasterScenario,
    validation: dict[str, Any],
    planner: dict[str, Any],
    prediction: dict[str, Any],
    rag: dict[str, Any],
) -> dict[str, Any]:
    issues = list(validation.get("issues", [])) if isinstance(validation.get("issues"), list) else []
    warnings = list(validation.get("warnings", [])) if isinstance(validation.get("warnings"), list) else []
    if issues:
        recommendation = "Proceed only with human review after resolving validation blockers."
    elif warnings:
        recommendation = "Proceed with the plan while actively tracking validation warnings."
    else:
        recommendation = "Proceed with the evidence-backed emergency action plan."

    phases = planner.get("phases", [])
    prioritized_actions = []
    if isinstance(phases, list):
        for phase in phases[:5]:
            if isinstance(phase, dict):
                prioritized_actions.append(
                    {
                        "phase": str(phase.get("phase", "response")),
                        "action": str(phase.get("action", "")),
                    }
                )
    if not prioritized_actions:
        prioritized_actions = [
            {"phase": "immediate", "action": f"Activate emergency coordination for {scenario_place(scenario)}."},
            {"phase": scenario.time_horizon, "action": "Revalidate predictions and resource capacity."},
        ]

    return {
        "executive_recommendation": recommendation,
        "evidence_citations": evidence_citations(rag, limit=5),
        "prioritized_actions": prioritized_actions,
        "model_outputs": prediction,
        "validation_results": validation,
        "unresolved_risks": issues + warnings,
        "generation_mode": "deterministic_fallback",
    }
