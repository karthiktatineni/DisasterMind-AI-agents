"""Four-agent native tool-calling Coordinator Orchestrator (Task 4)."""

from __future__ import annotations

import time
from typing import Any, Callable
from uuid import uuid4

import httpx

from backend.app.config import Settings, get_settings
from backend.app.orchestration.agent_runner import AgentSpec, run_agent_loop
from backend.app.orchestration.models import (
    AGENT_SEQUENCE,
    AgentName,
    AgentTraceRecord,
    Briefing,
    ConflictItem,
    EvacuationSummary,
    HumanReviewInfo,
    LogisticsSummary,
    OrchestrationRunResult,
    QuantityItem,
    RiskSummary,
    SharedState,
    SSEEventRecord,
    TokenUsageSummary,
)
from backend.app.orchestration.verifier import InAppVerifier
from backend.app.schemas import DisasterScenario, ScenarioIntakeRequest
from backend.app.services.historical_stats import HistoricalStatsService
from backend.app.services.llm_client import NativeToolLLMClient
from backend.app.services.retrieval_service import RetrievalService
from backend.app.tools.domain_tools import DelegateFocusArgs, build_domain_tools
from backend.app.tools.registry import ToolRegistry, ToolSpec

DEFAULT_DISCLAIMERS = [
    (
        "Scenario planning briefing only: quantities combine verified tool observations, "
        "EM-DAT historical records, and Sphere 2018 planning standards where applicable."
    ),
    "Decision support output does not authorize emergency orders without qualified human sign-off.",
]

AGENT_SPECS: dict[AgentName, AgentSpec] = {
    "coordinator": AgentSpec(
        name="coordinator",
        prompt_file="coordinator.md",
        allow_list=(
            "search_precedents",
            "geocode_region",
            "run_risk_agent",
            "run_logistics_agent",
            "run_evacuation_agent",
        ),
        submission_tool_name="submit_coordinator_report",
        submission_description="Submit final Coordinator synthesis after delegating to specialists.",
        temperature=0.0,
        seed=42,
        max_tokens=3200,
        max_tool_rounds=5,
        max_steps=8,
        max_token_budget=36000,
        wall_clock_cap_s=420.0,
        force_submit_on_required_ok=False,
    ),
    "risk": AgentSpec(
        name="risk",
        prompt_file="risk.md",
        allow_list=("search_precedents", "get_historical_stats", "geocode_region", "get_weather"),
        submission_tool_name="submit_risk_report",
        submission_description="Submit verified Risk & Premise assessment grounded in tool outputs.",
        temperature=0.0,
        seed=42,
        max_tokens=3200,
        max_tool_rounds=3,
        max_steps=6,
        max_token_budget=24000,
        wall_clock_cap_s=150.0,
        force_submit_on_required_ok=True,
    ),
    "logistics": AgentSpec(
        name="logistics",
        prompt_file="logistics.md",
        allow_list=("search_precedents", "get_historical_stats", "compute_logistics_requirements"),
        submission_tool_name="submit_logistics_report",
        submission_description="Submit verified Logistics capacity/standards assessment with provenance.",
        temperature=0.0,
        seed=42,
        max_tokens=3200,
        max_tool_rounds=3,
        max_steps=6,
        max_token_budget=24000,
        wall_clock_cap_s=150.0,
        force_submit_on_required_ok=True,
    ),
    "evacuation": AgentSpec(
        name="evacuation",
        prompt_file="evacuation.md",
        allow_list=("search_precedents", "get_historical_stats", "get_weather"),
        submission_tool_name="submit_evacuation_report",
        submission_description="Submit verified Evacuation recommendation grounded in precedent and Risk.",
        temperature=0.0,
        seed=42,
        max_tokens=3200,
        max_tool_rounds=3,
        max_steps=6,
        max_token_budget=24000,
        wall_clock_cap_s=150.0,
        force_submit_on_required_ok=True,
    ),
}


def _detect_conflicts(
    risk: RiskSummary,
    evacuation: EvacuationSummary,
) -> list[ConflictItem]:
    conflicts: list[ConflictItem] = []
    if risk.level in ("low", "insufficient_data") and evacuation.recommendation in (
        "voluntary_evacuation",
        "mandatory_evacuation",
    ):
        conflicts.append(
            ConflictItem(
                kind="risk_evacuation_mismatch",
                description=(
                    f"Evacuation recommends '{evacuation.recommendation}' while Risk level is '{risk.level}'."
                ),
                risk_level=risk.level,
                evacuation_recommendation=evacuation.recommendation,
            )
        )
    elif risk.level in ("high", "extreme") and evacuation.recommendation == "no_evacuation":
        conflicts.append(
            ConflictItem(
                kind="risk_evacuation_mismatch",
                description=(
                    f"Risk level is '{risk.level}' yet Evacuation recommends '{evacuation.recommendation}'."
                ),
                risk_level=risk.level,
                evacuation_recommendation=evacuation.recommendation,
            )
        )
    if not risk.premise_valid and evacuation.recommendation not in ("no_evacuation", "insufficient_data"):
        conflicts.append(
            ConflictItem(
                kind="invalid_premise_evacuation",
                description="Scenario physical premise was flagged invalid, yet positive evacuation was recommended.",
                risk_level=risk.level,
                evacuation_recommendation=evacuation.recommendation,
            )
        )
    return conflicts


class DisasterMindOrchestrator:
    """Four-agent Coordinator Orchestrator enforcing native tool calls, SharedState verification, and SSE transitions."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        llm_client: NativeToolLLMClient | None = None,
        retrieval_service: RetrievalService | None = None,
        historical_stats_service: HistoricalStatsService | None = None,
        http_client: httpx.AsyncClient | None = None,
        verifier: InAppVerifier | None = None,
        agent_specs: dict[AgentName, AgentSpec] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._llm_client = llm_client
        self._retrieval_service = retrieval_service or RetrievalService(settings=self.settings)
        self._historical_stats_service = historical_stats_service
        self._http_client = http_client
        self.verifier = verifier or InAppVerifier()
        self.agent_specs = dict(agent_specs or AGENT_SPECS)

    def _get_historical_stats_service(self) -> HistoricalStatsService | None:
        if self._historical_stats_service is None:
            try:
                self._historical_stats_service = HistoricalStatsService()
            except FileNotFoundError:
                return None
        return self._historical_stats_service

    async def run(
        self,
        scenario: DisasterScenario | ScenarioIntakeRequest | dict[str, Any],
        *,
        run_id: str | None = None,
        on_event: Callable[[SSEEventRecord], None] | None = None,
        mode_label: str = "hypothetical_scenario",
    ) -> OrchestrationRunResult:
        started = time.monotonic()
        rid = run_id or (scenario.request_id if isinstance(scenario, DisasterScenario) else str(uuid4()))

        scenario_dict = (
            scenario.model_dump() if hasattr(scenario, "model_dump") else dict(scenario)
        )
        # Keep only explicit scenario fields without inventing missing quantities.
        user_payload = {k: v for k, v in scenario_dict.items() if v is not None and v != ""}
        user_payload.pop("data_sources", None)

        events: list[SSEEventRecord] = []

        def emit(ev: SSEEventRecord) -> None:
            events.append(ev)
            if on_event is not None:
                on_event(ev)

        stats_svc = self._get_historical_stats_service()
        dataset_version = (
            stats_svc.dataset_version if stats_svc is not None else "unknown"
        )
        state = SharedState(
            run_id=rid,
            dataset_version=dataset_version,
            scenario_inputs=user_payload,
        )

        emit(
            SSEEventRecord(
                event="run_started",
                run_id=rid,
                data={
                    "disaster_type": user_payload.get("disaster_type"),
                    "region": user_payload.get("region"),
                    "dataset_version": dataset_version,
                },
            )
        )

        owns_http = self._http_client is None
        http_client = self._http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(15.0, connect=10.0),
            headers={"User-Agent": "DisasterMindAI/0.2"},
        )
        owns_llm = self._llm_client is None
        try:
            llm_client = self._llm_client or NativeToolLLMClient(settings=self.settings, http_client=http_client)
        except Exception as exc:
            if owns_http:
                await http_client.aclose()
            emit(
                SSEEventRecord(
                    event="run_failed",
                    run_id=rid,
                    data={"error": str(exc)},
                )
            )
            briefing = self._assemble_briefing(
                state=state,
                emit=emit,
                label=mode_label,
                extra_gap=f"LLM configuration error: {exc}",
            )
            return OrchestrationRunResult(
                briefing=briefing,
                agent_traces=[],
                events=events,
                wall_time_ms=round((time.monotonic() - started) * 1000.0, 2),
            )

        sub_traces: dict[AgentName, AgentTraceRecord] = {}
        try:
            domain_specs = build_domain_tools(
                retrieval_service=self._retrieval_service,
                historical_stats_service=stats_svc,
                http_client=http_client,
            )

            run_deadline = started + 450.0
            run_max_tokens = 120000

            # Seed resolved_geography immediately if user already supplied coordinates
            if user_payload.get("latitude") is not None and user_payload.get("longitude") is not None:
                state.resolved_geography = {
                    "latitude": user_payload["latitude"],
                    "longitude": user_payload["longitude"],
                    "country": user_payload.get("country"),
                    "resolved_location": user_payload.get("region"),
                    "population": user_payload.get("population"),
                    "provenance": "assumption supplied by user",
                }

            async def _delegate(target: AgentName, args: dict[str, Any], call_id: str) -> tuple[str, dict[str, Any], str | None]:
                if target in state.verified_reports:
                    return (
                        "ok",
                        {
                            "call_id": call_id,
                            "agent": target,
                            "status": "verified",
                            "already_provided": True,
                            "report": state.verified_reports[target].output,
                            "retrieved_record_ids": list(state.retrieved_records.keys()),
                        },
                        None,
                    )
                if target in state.budget_exhausted_agents or target in state.abstained_agents:
                    return (
                        "error",
                        {"call_id": call_id, "agent": target, "blocked": True},
                        f"Re-delegation blocked: '{target}' already abstained or exhausted its budget.",
                    )
                if target == "evacuation" and state.get_verified_risk() is None:
                    return (
                        "error",
                        {"precondition_failed": "risk_required", "call_id": call_id},
                        "Precondition failed: Risk agent must complete and be verified before Evacuation agent runs.",
                    )

                attempts = state.delegation_attempts.get(target, 0) + 1
                state.delegation_attempts[target] = attempts
                if attempts > 2:
                    return (
                        "error",
                        {"call_id": call_id, "agent": target, "blocked": True},
                        f"Delegation limit reached for '{target}'.",
                    )

                sub_context: dict[str, Any] = {
                    "scenario": user_payload,
                    "focus_question": args["focus_question"],
                }
                if state.resolved_geography:
                    sub_context["resolved_geography"] = state.resolved_geography
                if target == "logistics":
                    sub_context["verified_risk"] = (
                        state.verified_reports["risk"].output if "risk" in state.verified_reports else None
                    )
                    sub_context["retrieved_record_ids"] = list(state.retrieved_records.keys())
                elif target == "evacuation":
                    sub_context["verified_risk"] = state.verified_reports["risk"].output
                    sub_context["retrieved_record_ids"] = list(state.retrieved_records.keys())
                    sub_context["verified_logistics"] = (
                        state.verified_reports["logistics"].output if "logistics" in state.verified_reports else None
                    )

                trace = await run_agent_loop(
                    spec=self.agent_specs[target],
                    user_payload=sub_context,
                    state=state,
                    llm_client=llm_client,
                    registry=registry,
                    verifier=self.verifier,
                    emit=emit,
                    run_deadline=run_deadline,
                    run_max_tokens=run_max_tokens,
                )
                sub_traces[target] = trace
                if trace.status == "verified" and target in state.verified_reports:
                    return (
                        "ok",
                        {
                            "call_id": call_id,
                            "agent": target,
                            "status": "verified",
                            "report": state.verified_reports[target].output,
                            "retrieved_record_ids": list(state.retrieved_records.keys()),
                        },
                        None,
                    )
                reason = state.abstentions[-1].reason if state.abstentions else "Agent abstained."
                return (
                    "no_data",
                    {
                        "call_id": call_id,
                        "agent": target,
                        "status": "abstained",
                        "reason": reason,
                    },
                    reason,
                )

            async def _run_risk(args: dict[str, Any], call_id: str) -> tuple[str, dict[str, Any], str | None]:
                return await _delegate("risk", args, call_id)

            async def _run_logistics(args: dict[str, Any], call_id: str) -> tuple[str, dict[str, Any], str | None]:
                return await _delegate("logistics", args, call_id)

            async def _run_evacuation(args: dict[str, Any], call_id: str) -> tuple[str, dict[str, Any], str | None]:
                return await _delegate("evacuation", args, call_id)

            delegation_specs = [
                ToolSpec(
                    name="run_risk_agent",
                    description="Delegate risk & premise verification to the Risk Agent.",
                    args_model=DelegateFocusArgs,
                    handler=_run_risk,
                    timeout_s=180.0,
                ),
                ToolSpec(
                    name="run_logistics_agent",
                    description="Delegate humanitarian capacity & WHO/Sphere water analysis to the Logistics Agent.",
                    args_model=DelegateFocusArgs,
                    handler=_run_logistics,
                    timeout_s=180.0,
                ),
                ToolSpec(
                    name="run_evacuation_agent",
                    description="Delegate evacuation posture decision to the Evacuation Agent (requires verified Risk report).",
                    args_model=DelegateFocusArgs,
                    handler=_run_evacuation,
                    timeout_s=180.0,
                ),
            ]
            registry = ToolRegistry(domain_specs + delegation_specs)

            coord_trace = await run_agent_loop(
                spec=self.agent_specs["coordinator"],
                user_payload={"scenario": user_payload},
                state=state,
                llm_client=llm_client,
                registry=registry,
                verifier=self.verifier,
                emit=emit,
                run_deadline=run_deadline,
                run_max_tokens=run_max_tokens,
            )

            ordered_traces: list[AgentTraceRecord] = [coord_trace]
            for name in ("risk", "logistics", "evacuation"):
                if name in sub_traces:
                    ordered_traces.append(sub_traces[name])

            briefing = self._assemble_briefing(
                state=state,
                emit=emit,
                label=mode_label,
            )
            emit(
                SSEEventRecord(
                    event="briefing_ready",
                    run_id=rid,
                    data={
                        "status": briefing.status,
                        "conflicts": len(briefing.conflicts),
                        "briefing": briefing.model_dump(),
                    },
                )
            )

            total_prompt = sum(t.usage.prompt_tokens for t in ordered_traces)
            total_comp = sum(t.usage.completion_tokens for t in ordered_traces)
            total_reason = sum(t.usage.reasoning_tokens for t in ordered_traces)
            total_toks = sum(t.usage.total_tokens for t in ordered_traces)
            llm_calls = sum(len(t.steps) for t in ordered_traces)

            return OrchestrationRunResult(
                briefing=briefing,
                agent_traces=ordered_traces,
                events=events,
                total_llm_calls=llm_calls,
                total_tool_calls=len(state.tool_history),
                token_usage=TokenUsageSummary(
                    prompt_tokens=total_prompt,
                    completion_tokens=total_comp,
                    reasoning_tokens=total_reason,
                    total_tokens=total_toks,
                ),
                wall_time_ms=round((time.monotonic() - started) * 1000.0, 2),
            )
        finally:
            if owns_llm and hasattr(llm_client, "aclose"):
                await llm_client.aclose()
            if owns_http:
                await http_client.aclose()

    def _assemble_briefing(
        self,
        *,
        state: SharedState,
        emit: Callable[[SSEEventRecord], None],
        label: str,
        extra_gap: str | None = None,
    ) -> Briefing:
        risk_out = state.get_verified_risk()
        log_out = state.get_verified_logistics()
        evac_out = state.get_verified_evacuation()
        coord_out = state.get_verified_coordinator()

        risk_summary = (
            RiskSummary(
                level=risk_out.risk_level,
                premise_valid=risk_out.premise_valid,
                premise_issue=risk_out.premise_issue,
                summary=risk_out.hazard_summary,
                verified_by_agent="risk",
            )
            if risk_out is not None
            else RiskSummary(
                level="insufficient_data",
                premise_valid=True,
                summary="Risk agent abstained or did not verify.",
                verified_by_agent=None,
            )
        )

        evac_summary = (
            EvacuationSummary(
                recommendation=evac_out.recommendation,
                target_headcount=evac_out.target_headcount,
                target_headcount_provenance=evac_out.target_headcount_provenance,
                trigger_criteria=list(evac_out.trigger_criteria),
                priority_zones=list(evac_out.priority_zones),
                verified_by_agent="evacuation",
            )
            if evac_out is not None
            else EvacuationSummary(
                recommendation="insufficient_data",
                verified_by_agent=None,
            )
        )

        log_summary = (
            LogisticsSummary(
                resource_adequacy=log_out.resource_adequacy,
                capacity_comparisons=list(log_out.capacity_comparisons),
                supply_recommendations=list(log_out.supply_recommendations),
                verified_by_agent="logistics",
            )
            if log_out is not None
            else None
        )

        conflicts = _detect_conflicts(risk_summary, evac_summary)
        for conf in conflicts:
            state.conflicts.append(conf)
            emit(
                SSEEventRecord(
                    event="conflict_flagged",
                    run_id=state.run_id,
                    data=conf.model_dump(),
                )
            )

        # Deduplicate cited EM-DAT records across verified outputs only
        seen_cits: set[str] = set()
        citations = []
        for out in (coord_out, risk_out, log_out, evac_out):
            if out is None:
                continue
            for rid in out.cited_record_ids:
                if rid in state.retrieved_records and rid not in seen_cits:
                    seen_cits.add(rid)
                    citations.append(state.retrieved_records[rid])

        # Collect verified QuantityItems
        quantities: list[QuantityItem] = []
        seen_q: set[tuple[str, str, str]] = set()
        for out in (risk_out, log_out, evac_out):
            if out is None:
                continue
            for q in out.quantities:
                key = (q.name, str(q.value), q.provenance)
                if key not in seen_q:
                    seen_q.add(key)
                    quantities.append(q)

        # Collect data gaps honestly
        data_gaps: list[str] = []
        if extra_gap:
            data_gaps.append(extra_gap)
        for out in (coord_out, risk_out, log_out, evac_out):
            if out is None:
                continue
            for gap in out.data_gaps:
                if gap not in data_gaps:
                    data_gaps.append(gap)
        for abst in state.abstentions:
            msg = f"{abst.agent} abstained: {abst.reason}"
            if msg not in data_gaps:
                data_gaps.append(msg)

        specialists_verified = [a for a in ("risk", "logistics", "evacuation") if a in state.verified_reports]
        if not specialists_verified or (
            risk_summary.level == "insufficient_data"
            and evac_summary.recommendation in ("insufficient_data", "no_evacuation")
            and not citations
        ):
            status = "insufficient_data"
        elif all(a in state.verified_reports for a in AGENT_SEQUENCE) and not state.abstentions:
            status = "complete"
        else:
            status = "partial"

        review_reasons: list[str] = []
        if evac_summary.recommendation in ("standby_advisory", "voluntary_evacuation", "mandatory_evacuation"):
            review_reasons.append(
                f"Evacuation action '{evac_summary.recommendation}' requires mandatory human approval."
            )
        if conflicts:
            review_reasons.append("Risk and Evacuation assessments disagree.")
        if state.abstentions:
            review_reasons.append("One or more agents abstained cleanly.")
        if not risk_summary.premise_valid:
            review_reasons.append(f"Invalid physical premise: {risk_summary.premise_issue}")
        if status != "complete":
            review_reasons.append(f"Briefing completed with status '{status}'.")

        human_required = len(review_reasons) > 0 or (
            evac_summary.recommendation not in ("no_evacuation", "insufficient_data")
        )

        label_lit: Any = (
            "historical_precedent"
            if label == "historical_precedent"
            else "hypothetical_scenario"
        )

        exec_summary = (
            coord_out.executive_summary
            if coord_out is not None
            else f"[{status}] Built deterministically from verified agents: {specialists_verified or ['none']}."
        )
        action_plan = list(coord_out.action_plan) if coord_out is not None else []

        return Briefing(
            run_id=state.run_id,
            status=status,
            risk=risk_summary,
            evacuation_recommendation=evac_summary,
            logistics_summary=log_summary,
            quantities=quantities,
            citations=citations,
            data_gaps=data_gaps,
            abstentions=list(state.abstentions),
            conflicts=conflicts,
            human_review=HumanReviewInfo(required=human_required, reasons=review_reasons),
            disclaimers=list(DEFAULT_DISCLAIMERS),
            label=label_lit,
            model_ids=list(state.model_ids) or [self.settings.nvidia_model],
            dataset_version=state.dataset_version,
            executive_summary=exec_summary,
            action_plan=action_plan,
        )
