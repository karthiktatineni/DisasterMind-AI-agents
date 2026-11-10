"""Strict in-app verifier gating every write to SharedState (Task 4)."""

from __future__ import annotations

from dataclasses import dataclass
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
    "standard:sphere_2018_wash_15l",
    "standard:sphere_2018_shelter_3_5m2",
    "standard:sphere_2018_food_2100kcal",
}


@dataclass(frozen=True)
class VerificationResult:
    valid: bool
    errors: list[str]
    validated_model: BaseModel | None = None


def is_valid_provenance(prov: str, state: SharedState) -> bool:
    norm = prov.strip()
    if not norm:
        return False
    low = norm.lower()
    if norm in state.tool_calls:
        return True
    if norm in state.retrieved_records:
        return True
    if low == "assumption supplied by user" or low.startswith("assumption supplied by user:"):
        return True
    if low in ALLOWED_STANDARD_PROVENANCES or low.startswith("standard:sphere_2018_"):
        return True
    # Composite standard + call_id provenance (e.g., "call_1 + standard:sphere_2018_wash_15L")
    for part in [p.strip() for p in norm.replace(";", "+").split("+")]:
        pl = part.lower()
        if part in state.tool_calls or part in state.retrieved_records:
            return True
        if pl.startswith("standard:sphere_2018_"):
            return True
    return False


class InAppVerifier:
    """Validates typed output, required tools, citations, provenance, and inventory honesty."""

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

        # 2. All cited EM-DAT record IDs must exist in retrieved_records from this run
        for rid in getattr(validated, "cited_record_ids", []):
            if rid not in state.retrieved_records:
                errors.append(f"Citation '{rid}' does not exist in retrieved_records for this run.")

        # 3. Validate QuantityItem provenance
        for item in getattr(validated, "quantities", []):
            if not is_valid_provenance(item.provenance, state):
                errors.append(
                    f"Quantity '{item.name}' ({item.value} {item.unit}) has invalid provenance '{item.provenance}'."
                )

        # 4. Agent-specific strict invariants
        if isinstance(validated, RiskAgentOutput):
            if not validated.premise_valid and not (validated.premise_issue and validated.premise_issue.strip()):
                errors.append("When premise_valid=false, premise_issue must explain the physical impossibility.")
            if not validated.premise_valid and validated.cited_record_ids:
                errors.append("Physically invalid premise must not cite supporting precedents as validating evidence.")

        if isinstance(validated, LogisticsAgentOutput):
            for comp in validated.capacity_comparisons:
                if not is_valid_provenance(comp.provenance, state):
                    errors.append(
                        f"Capacity comparison '{comp.resource}' has invalid provenance '{comp.provenance}'."
                    )
                if comp.available_status == "not_provided" and comp.available is not None:
                    errors.append(f"'{comp.resource}' marked not_provided must have available=null.")
                if comp.available_status == "provided_by_user" and comp.available is None:
                    errors.append(f"'{comp.resource}' marked provided_by_user must carry an explicit value.")

        if isinstance(validated, EvacuationAgentOutput):
            if validated.target_headcount is not None:
                prov = validated.target_headcount_provenance or ""
                if not is_valid_provenance(prov, state):
                    errors.append(
                        f"Evacuation target_headcount={validated.target_headcount} missing valid provenance ('{prov}')."
                    )

        return VerificationResult(
            valid=(len(errors) == 0),
            errors=errors,
            validated_model=validated if len(errors) == 0 else None,
        )
