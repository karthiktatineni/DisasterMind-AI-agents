"""Strict in-app verifier enforcing numeric grounding, provenance, citation claim-support, and premise validity (Tasks 4 & 5)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from backend.app.orchestration.models import (
    AgentName,
    CoordinatorAgentOutput,
    EvacuationAgentOutput,
    LogisticsAgentOutput,
    RiskAgentOutput,
    SharedState,
)

AGENT_REQUIRED_TOOLS: dict[AgentName, tuple[str, ...]] = {
    "coordinator": ("run_risk_agent",),
    "risk": ("search_precedents",),
    "logistics": ("get_historical_stats",),
    "evacuation": ("search_precedents",),
}

AGENT_OUTPUT_MODELS: dict[AgentName, type[BaseModel]] = {
    "coordinator": CoordinatorAgentOutput,
    "risk": RiskAgentOutput,
    "logistics": LogisticsAgentOutput,
    "evacuation": EvacuationAgentOutput,
}

ALLOWED_STANDARD_PROVENANCES = {
    "standard",
    "standard:water_litres_per_person_per_day",
}

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STANDARDS_PATH = PROJECT_ROOT / "config" / "planning_standards.json"

_URL_RE = re.compile(r"https?://[^\s)\],\"']+", re.IGNORECASE)
_SCALED_NUM_RE = re.compile(
    r"(?<![A-Za-z0-9_.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(k\b|thousand\b|million\b|billion\b)",
    re.IGNORECASE,
)
_NUM_RE = re.compile(
    r"(?<![A-Za-z0-9_.])([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(?=(?:[kK]\b|%|°|\s|$|[,;:.!?)/+-]|['\"]|[A-Za-z]))"
)
_IMPOSSIBLE_CATEGORY_RE = re.compile(r"\b(?:category|cat\.?)\s*(?:[6-9]|[1-9]\d+)\b", re.IGNORECASE)
_IMPOSSIBLE_MAG_RE = re.compile(
    r"\b(?:magnitude|mw|m_w|richter)\s*(?:of\s*)?(?:1[0-9]|[2-9]\d)(?:\.\d+)?\b", re.IGNORECASE
)
_IMPOSSIBLE_HUMIDITY_RE = re.compile(
    r"\b(?:humidity|relative_humidity|rh)\s*(?:of|at|=|:|>)?\s*(?:10[1-9]|1[1-9]\d|[2-9]\d{2})\s*%",
    re.IGNORECASE,
)
_IMPOSSIBLE_WIND_RE = re.compile(
    r"\b([5-9]\d{2}|[1-9]\d{3,})\s*(?:km/h|kph|kmh)\b", re.IGNORECASE
)

_HAZARD_FAMILIES: dict[str, set[str]] = {
    "storm": {"storm", "cyclone", "tropical cyclone", "hurricane", "typhoon", "tornado", "gale"},
    "flood": {"flood", "flooding", "flash flood", "riverine flood", "coastal flood", "inundation"},
    "earthquake": {"earthquake", "seismic", "quake", "tsunami", "tremor"},
    "wildfire": {"wildfire", "bushfire", "forest fire", "fire"},
    "drought": {"drought", "arid"},
    "landslide": {"landslide", "mudslide", "mass movement"},
    "volcanic": {"volcano", "volcanic", "volcanic activity", "eruption"},
}


@dataclass(frozen=True)
class VerificationResult:
    valid: bool
    errors: list[str]
    validated_model: BaseModel | None = None


@lru_cache(maxsize=8)
def _read_standards_json(target_str: str) -> str:
    return Path(target_str).read_text(encoding="utf-8")


def load_standard_numbers(path: Path | None = None) -> set[float]:
    """Load strictly verified numeric planning constants from config/planning_standards.json."""
    target = path or STANDARDS_PATH
    if not target.exists():
        return {15.0}
    doc = json.loads(_read_standards_json(str(target)))
    nums: set[float] = set()
    for entry in (doc.get("standards") or {}).values():
        if isinstance(entry, dict) and isinstance(entry.get("value"), (int, float)) and not isinstance(entry.get("value"), bool):
            nums.add(float(entry["value"]))
    return nums


def extract_numbers_from_text(text: str, *, masked_tokens: set[str] | None = None) -> list[float]:
    """Extract numeric values from prose while ignoring verified identifier tokens and URLs."""
    if not text or not isinstance(text, str):
        return []
    cleaned = _URL_RE.sub(" ", text)
    if masked_tokens:
        for tok in sorted((t for t in masked_tokens if t and len(t) >= 2), key=len, reverse=True):
            cleaned = cleaned.replace(tok, " ")

    out: list[float] = []
    consumed_spans: list[tuple[int, int]] = []
    multipliers = {"k": 1e3, "thousand": 1e3, "million": 1e6, "billion": 1e9}

    for m in _SCALED_NUM_RE.finditer(cleaned):
        raw = m.group(1).replace(",", "")
        unit = m.group(2).lower()
        out.append(float(raw) * multipliers[unit])
        consumed_spans.append(m.span())

    for m in _NUM_RE.finditer(cleaned):
        start, end = m.span()
        if any(s <= start and end <= e for s, e in consumed_spans):
            continue
        # Only parse numeric tokens followed by punctuation/space/units — skip random model tokens like '550b'
        suffix = cleaned[m.end() : m.end() + 4].lower()
        if m.end() < len(cleaned) and cleaned[m.end()].isalpha():
            if not (suffix.startswith(("k", "km", "l/", "l ", "c ", "c.", "h ", "hr", "h.", "m ", "m/", "s "))):
                # Allow 1-char unit only when exact unit (e.g. '24h', '15L')
                next_char = cleaned[m.end()]
                after_char = cleaned[m.end() + 1 : m.end() + 2]
                if not (next_char in "LlHhMmSsKkCc" and (not after_char or not after_char.isalnum())):
                    continue

        raw = m.group(1).replace(",", "")
        # Range hyphen like "1-5": leading '-' preceded by a digit should be positive 5, not -5
        if raw.startswith("-") and m.start() > 0 and cleaned[m.start() - 1].isdigit():
            raw = raw[1:]
        try:
            out.append(float(raw))
        except ValueError:
            continue
    return out


def collect_numbers_from_value(obj: Any, *, masked_tokens: set[str] | None = None) -> set[float]:
    """Recursively collect all numeric values (skipping bool) from structured data."""
    nums: set[float] = set()
    if isinstance(obj, bool) or obj is None:
        return nums
    if isinstance(obj, (int, float)):
        if math.isfinite(float(obj)):
            nums.add(float(obj))
        return nums
    if isinstance(obj, str):
        nums.update(extract_numbers_from_text(obj, masked_tokens=masked_tokens))
        return nums
    if isinstance(obj, dict):
        for v in obj.values():
            nums.update(collect_numbers_from_value(v, masked_tokens=masked_tokens))
        return nums
    if isinstance(obj, (list, tuple, set)):
        for item in obj:
            nums.update(collect_numbers_from_value(item, masked_tokens=masked_tokens))
        return nums
    return nums


def _build_masked_tokens(state: SharedState) -> set[str]:
    tokens: set[str] = {state.run_id, state.dataset_version, *state.model_ids, *state.retrieved_records.keys(), *state.tool_calls.keys()}
    return {t for t in tokens if t and len(t) >= 2}


def build_user_input_numbers(state: SharedState) -> set[float]:
    """Build exact set of numbers supplied directly in user scenario inputs."""
    masked = _build_masked_tokens(state)
    filtered_inputs = {k: v for k, v in state.scenario_inputs.items() if k != "request_id"}
    return collect_numbers_from_value(filtered_inputs, masked_tokens=masked)


def build_allowed_number_set(
    state: SharedState,
    *,
    standards_numbers: set[float] | None = None,
) -> set[float]:
    """Build complete allowed-number set from user input, ok/no_data tool results, retrieved records, and standards."""
    masked = _build_masked_tokens(state)
    allowed: set[float] = set()
    allowed.update(build_user_input_numbers(state))
    allowed.update(load_standard_numbers() if standards_numbers is None else standards_numbers)

    for env in state.tool_history:
        if env.status in ("ok", "no_data"):
            allowed.update(collect_numbers_from_value(env.data, masked_tokens=masked))

    for cit in state.retrieved_records.values():
        allowed.update(collect_numbers_from_value(cit.model_dump(), masked_tokens=masked))

    return allowed


def number_in_allowed(val: int | float, allowed: set[float]) -> bool:
    if isinstance(val, bool) or not math.isfinite(float(val)):
        return False
    target = float(val)
    return any(math.isclose(target, cand, rel_tol=1e-6, abs_tol=1e-4) for cand in allowed)


def detect_physical_impossibility(scenario_inputs: dict[str, Any]) -> str | None:
    """Return a clear explanation if scenario inputs contain physically impossible parameters."""
    lat = scenario_inputs.get("latitude")
    lon = scenario_inputs.get("longitude")
    hum = scenario_inputs.get("humidity")
    wind = scenario_inputs.get("wind_speed")
    elev = scenario_inputs.get("elevation")

    if isinstance(lat, (int, float)) and not (-90.0 <= float(lat) <= 90.0):
        return f"Latitude {lat} is outside Earth coordinate bounds [-90, 90]."
    if isinstance(lon, (int, float)) and not (-180.0 <= float(lon) <= 180.0):
        return f"Longitude {lon} is outside Earth coordinate bounds [-180, 180]."
    if isinstance(hum, (int, float)) and not (0.0 <= float(hum) <= 100.0):
        return f"Relative humidity {hum}% exceeds physical saturation bounds [0, 100]%."
    if isinstance(wind, (int, float)) and float(wind) > 500.0:
        return f"Sustained wind speed {wind} km/h exceeds plausible planetary cyclone physics."
    if isinstance(elev, (int, float)) and not (-500.0 <= float(elev) <= 9000.0):
        return f"Elevation {elev} m is outside terrestrial surface bounds [-500, 9000] m."

    combined_text = " ".join(
        str(scenario_inputs.get(k) or "")
        for k in ("disaster_type", "region", "time_horizon", "notes")
    )
    if _IMPOSSIBLE_CATEGORY_RE.search(combined_text):
        return "Scenario specifies tropical cyclone Category 6 or higher, which does not exist on the Saffir-Simpson scale."
    if _IMPOSSIBLE_MAG_RE.search(combined_text):
        return "Scenario specifies earthquake magnitude 10 or higher, exceeding terrestrial fault-rupture physics."
    if _IMPOSSIBLE_HUMIDITY_RE.search(combined_text):
        return "Scenario specifies relative humidity above 100% saturation."
    if _IMPOSSIBLE_WIND_RE.search(combined_text):
        return "Scenario specifies wind speed above 500 km/h, exceeding terrestrial storm bounds."
    return None


def _hazard_family(text: str | None) -> str | None:
    if not text:
        return None
    low = text.strip().lower()
    for fam, terms in _HAZARD_FAMILIES.items():
        if any(t in low for t in terms):
            return fam
    return None


def _numbers_authorized_by_provenance(
    prov: str,
    state: SharedState,
    *,
    standards_numbers: set[float],
) -> set[float]:
    masked = _build_masked_tokens(state)
    user_nums = build_user_input_numbers(state)
    parts = [p.strip() for p in prov.strip().replace(";", "+").split("+") if p.strip()]
    nums: set[float] = set()
    for part in parts:
        pl = part.lower()
        if part in state.tool_calls and state.tool_calls[part].status in ("ok", "no_data"):
            nums.update(collect_numbers_from_value(state.tool_calls[part].data, masked_tokens=masked))
        elif part in state.retrieved_records:
            nums.update(collect_numbers_from_value(state.retrieved_records[part].model_dump(), masked_tokens=masked))
        elif pl == "assumption supplied by user" or pl.startswith("assumption supplied by user"):
            nums.update(user_nums)
        elif pl in ALLOWED_STANDARD_PROVENANCES or pl.startswith("standard:water_litres_per_person_per_day"):
            nums.update(standards_numbers)
    return nums


def is_valid_provenance(prov: str, state: SharedState) -> bool:
    norm = prov.strip()
    if not norm:
        return False
    parts = [p.strip() for p in norm.replace(";", "+").split("+") if p.strip()]
    if not parts:
        return False
    for part in parts:
        pl = part.lower()
        if part in state.tool_calls and state.tool_calls[part].status in ("ok", "no_data"):
            continue
        if part in state.retrieved_records:
            continue
        if pl == "assumption supplied by user" or pl.startswith("assumption supplied by user"):
            continue
        if pl in ALLOWED_STANDARD_PROVENANCES or pl.startswith("standard:water_litres_per_person_per_day"):
            continue
        return False
    return True


def _extract_narrative_strings(validated: BaseModel) -> list[tuple[str, str]]:
    fields = (
        "hazard_summary",
        "premise_issue",
        "executive_summary",
        "supply_recommendations",
        "trigger_criteria",
        "priority_zones",
        "action_plan",
        "data_gaps",
    )
    items: list[tuple[str, str]] = []
    for fname in fields:
        val = getattr(validated, fname, None)
        if isinstance(val, str) and val.strip():
            items.append((fname, val))
        elif isinstance(val, list):
            for idx, entry in enumerate(val):
                if isinstance(entry, str) and entry.strip():
                    items.append((f"{fname}[{idx}]", entry))
    return items


class InAppVerifier:
    """Validates typed output, required tools, citations & claim-support, strict numeric grounding, and premises."""

    def __init__(self, *, threshold_used: float = 0.3112, standards_path: Path | None = None) -> None:
        self.threshold_used = threshold_used
        self.standards_numbers = load_standard_numbers(standards_path)

    def validate_tool_call_grounding(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        state: SharedState,
    ) -> str | None:
        """Reject ungrounded parameters before tool execution (e.g., invented population in logistics)."""
        if tool_name == "compute_logistics_requirements":
            pop = arguments.get("population")
            pop_prov = str(arguments.get("population_provenance") or "")
            if not is_valid_provenance(pop_prov, state):
                return (
                    f"Rejected compute_logistics_requirements: population_provenance '{pop_prov}' "
                    "is not an active tool call_id, retrieved record_id, or 'assumption supplied by user'."
                )
            authorized = _numbers_authorized_by_provenance(
                pop_prov, state, standards_numbers=self.standards_numbers
            )
            allowed = build_allowed_number_set(state, standards_numbers=self.standards_numbers)
            if pop is None or not (number_in_allowed(pop, authorized) and number_in_allowed(pop, allowed)):
                return (
                    f"Rejected compute_logistics_requirements: population={pop} is not grounded in '{pop_prov}' or user/tool inputs."
                )
            for opt_key in ("available_shelter_capacity", "available_water_litres_per_day"):
                opt_val = arguments.get(opt_key)
                if opt_val is not None and not number_in_allowed(opt_val, build_user_input_numbers(state)):
                    return (
                        f"Rejected compute_logistics_requirements: {opt_key}={opt_val} was not supplied by user input."
                    )
        return None

    def _verify_citations_support_claim(
        self,
        *,
        agent: AgentName,
        validated: BaseModel,
        state: SharedState,
        errors: list[str],
    ) -> None:
        cited_ids = list(getattr(validated, "cited_record_ids", []))
        verified_risk = state.get_verified_risk()
        premise_invalid = (isinstance(validated, RiskAgentOutput) and not validated.premise_valid) or (
            verified_risk is not None and not verified_risk.premise_valid
        )

        if premise_invalid and cited_ids:
            errors.append(
                "Physically invalid premise must not cite historical EM-DAT records as precedents (cited_record_ids must be [])."
            )
        elif not state.retrieved_records and cited_ids:
            errors.append(
                f"Cannot cite {cited_ids}: zero EM-DAT records met similarity threshold in this run."
            )
        else:
            scen_family = _hazard_family(str(state.scenario_inputs.get("disaster_type") or ""))
            scen_country = str(
                state.scenario_inputs.get("country")
                or state.resolved_geography.get("country")
                or ""
            ).strip().lower()

            for rid in cited_ids:
                if rid not in state.retrieved_records:
                    errors.append(f"Citation '{rid}' does not exist in retrieved_records for this run.")
                    continue
                cit = state.retrieved_records[rid]
                if float(cit.similarity) < self.threshold_used:
                    errors.append(
                        f"Citation '{rid}' similarity {cit.similarity} is below calibrated threshold {self.threshold_used}."
                    )
                cit_family = _hazard_family(cit.disaster_type)
                cit_country = (cit.country or "").strip().lower()
                hazard_match = bool(scen_family and cit_family and scen_family == cit_family)
                country_match = bool(scen_country and cit_country and (scen_country in cit_country or cit_country in scen_country))
                if (scen_family or scen_country) and not (hazard_match or country_match):
                    errors.append(
                        f"Citation '{rid}' ({cit.disaster_type}, {cit.country}) does not support scenario hazard/region."
                    )

        if isinstance(validated, RiskAgentOutput) and not state.retrieved_records and validated.risk_level != "insufficient_data":
            errors.append(
                "When no precedents match above calibrated threshold, Risk must report risk_level='insufficient_data'."
            )
        if (
            isinstance(validated, EvacuationAgentOutput)
            and (premise_invalid or not state.retrieved_records)
            and validated.recommendation not in ("insufficient_data", "no_evacuation")
        ):
            errors.append(
                f"Without valid precedent/premise, Evacuation cannot recommend '{validated.recommendation}'."
            )

    def verify_agent_output(
        self,
        *,
        agent: AgentName,
        raw_payload: dict[str, Any],
        agent_called_tools: list[str],
        state: SharedState,
    ) -> VerificationResult:
        errors: list[str] = []
        model_cls = AGENT_OUTPUT_MODELS[agent]
        try:
            validated = model_cls.model_validate(raw_payload)
        except ValidationError as exc:
            return VerificationResult(valid=False, errors=[f"Schema error: {exc.errors()}"])

        # 1. Required-tool rule: agent cannot skip its mandatory evidence tool
        for req_tool in AGENT_REQUIRED_TOOLS[agent]:
            if req_tool not in agent_called_tools:
                errors.append(
                    f"Required tool '{req_tool}' was skipped by agent '{agent}'. Execute it before submitting."
                )

        # 2. Citations must exist in retrieved_records and support the claim
        self._verify_citations_support_claim(agent=agent, validated=validated, state=state, errors=errors)

        # 3. Build strict allowed-number set from user input, tool results, retrieved records, and standards
        allowed_numbers = build_allowed_number_set(state, standards_numbers=self.standards_numbers)
        masked_tokens = _build_masked_tokens(state)

        # 4. Validate QuantityItem provenance AND exact numeric value grounding
        for item in getattr(validated, "quantities", []):
            if not is_valid_provenance(item.provenance, state):
                errors.append(
                    f"Quantity '{item.name}' ({item.value} {item.unit}) has invalid provenance '{item.provenance}'."
                )
                continue
            auth_nums = _numbers_authorized_by_provenance(
                item.provenance, state, standards_numbers=self.standards_numbers
            )
            if not (number_in_allowed(item.value, auth_nums) and number_in_allowed(item.value, allowed_numbers)):
                errors.append(
                    f"Quantity '{item.name}' value={item.value} is not grounded in provenance '{item.provenance}'."
                )

        # 5. Validate narrative prose contains zero ungrounded numbers
        for field_path, text in _extract_narrative_strings(validated):
            for num in extract_numbers_from_text(text, masked_tokens=masked_tokens):
                if not number_in_allowed(num, allowed_numbers):
                    errors.append(
                        f"Ungrounded number {num} in narrative field '{field_path}'. Only exact numbers from scenario inputs, verified tools, retrieved records, or 15 L/person/day standard are allowed."
                    )

        # 6. Agent-specific strict invariants & premise checks
        if isinstance(validated, RiskAgentOutput):
            impossible_reason = detect_physical_impossibility(state.scenario_inputs)
            if impossible_reason and validated.premise_valid:
                errors.append(
                    f"Physical impossibility detected ({impossible_reason}); set premise_valid=false, risk_level='insufficient_data', and cited_record_ids=[]."
                )
            if not validated.premise_valid and not (validated.premise_issue and validated.premise_issue.strip()):
                errors.append("When premise_valid=false, premise_issue must explain the physical impossibility.")

        if isinstance(validated, LogisticsAgentOutput):
            user_nums = build_user_input_numbers(state)
            for comp in validated.capacity_comparisons:
                if not is_valid_provenance(comp.provenance, state):
                    errors.append(
                        f"Capacity comparison '{comp.resource}' has invalid provenance '{comp.provenance}'."
                    )
                else:
                    auth_nums = _numbers_authorized_by_provenance(
                        comp.provenance, state, standards_numbers=self.standards_numbers
                    )
                    if not (number_in_allowed(comp.required, auth_nums) and number_in_allowed(comp.required, allowed_numbers)):
                        errors.append(
                            f"Capacity comparison '{comp.resource}' required={comp.required} is not grounded in '{comp.provenance}'."
                        )
                if comp.available_status == "not_provided":
                    if comp.available is not None:
                        errors.append(f"'{comp.resource}' marked not_provided must have available=null.")
                    if comp.deficit_or_surplus is not None:
                        errors.append(f"'{comp.resource}' marked not_provided must have deficit_or_surplus=null.")
                if comp.available_status == "provided_by_user":
                    if comp.available is None:
                        errors.append(f"'{comp.resource}' marked provided_by_user must carry an explicit value.")
                    elif not number_in_allowed(comp.available, user_nums):
                        errors.append(
                            f"'{comp.resource}' available={comp.available} was not supplied by user input."
                        )

        if isinstance(validated, EvacuationAgentOutput):
            if validated.target_headcount is not None:
                prov = validated.target_headcount_provenance or ""
                if not is_valid_provenance(prov, state):
                    errors.append(
                        f"Evacuation target_headcount={validated.target_headcount} missing valid provenance ('{prov}')."
                    )
                else:
                    auth_nums = _numbers_authorized_by_provenance(
                        prov, state, standards_numbers=self.standards_numbers
                    )
                    if not (
                        number_in_allowed(validated.target_headcount, auth_nums)
                        and number_in_allowed(validated.target_headcount, allowed_numbers)
                    ):
                        errors.append(
                            f"Evacuation target_headcount={validated.target_headcount} is not grounded in provenance '{prov}'."
                        )

        return VerificationResult(
            valid=(len(errors) == 0),
            errors=errors,
            validated_model=validated if len(errors) == 0 else None,
        )
