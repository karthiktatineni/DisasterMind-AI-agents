"""Strict typed Pydantic schemas (extra='forbid') for Task 4 4-agent orchestration and Briefing output."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

AGENT_SEQUENCE = ["coordinator", "risk", "logistics", "evacuation"]

AgentName = Literal["coordinator", "risk", "logistics", "evacuation"]
SSEEventType = Literal[
    "run_started",
    "agent_started",
    "tool_called",
    "tool_result",
    "agent_verified",
    "agent_rejected",
    "agent_abstained",
    "state_written",
    "conflict_flagged",
    "briefing_ready",
    "run_failed",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QuantityItem(_StrictModel):
    name: str = Field(min_length=1)
    value: int | float
    unit: str = Field(min_length=1)
    provenance: str = Field(min_length=1)
    inputs: list[str] = Field(default_factory=list)


class CapacityComparison(_StrictModel):
    resource: str = Field(min_length=1)
    required: int | float
    required_unit: str = Field(min_length=1)
    available: int | float | None = None
    available_status: Literal["provided_by_user", "not_provided"]
    deficit_or_surplus: int | float | None = None
    provenance: str = Field(min_length=1)
    inputs: list[str] = Field(default_factory=list)


class CitationItem(_StrictModel):
    """Allow-listed EM-DAT citation fields exposed to models and users (never raw search_text)."""

    record_id: str = Field(min_length=1)
    event_name: str = Field(min_length=1)
    disaster_type: str | None = None
    country: str | None = None
    region: str | None = None
    start_year: int | None = None
    total_deaths: float | None = None
    total_affected: float | None = None
    total_damage_usd_thousands: float | None = None
    similarity: float


class RiskAgentOutput(_StrictModel):
    risk_level: Literal["low", "moderate", "high", "extreme", "insufficient_data"]
    premise_valid: bool
    premise_issue: str | None = None
    hazard_summary: str = Field(min_length=1)
    cited_record_ids: list[str] = Field(default_factory=list)
    quantities: list[QuantityItem] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)


class LogisticsAgentOutput(_StrictModel):
    resource_adequacy: Literal["adequate", "constrained", "insufficient_inventory_data", "insufficient_data"]
    capacity_comparisons: list[CapacityComparison] = Field(default_factory=list)
    quantities: list[QuantityItem] = Field(default_factory=list)
    supply_recommendations: list[str] = Field(default_factory=list)
    cited_record_ids: list[str] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)


class EvacuationAgentOutput(_StrictModel):
    recommendation: Literal[
        "no_evacuation",
        "standby_advisory",
        "voluntary_evacuation",
        "mandatory_evacuation",
        "insufficient_data",
    ]
    requires_human_approval: bool | None = None
    target_headcount: int | None = Field(default=None, ge=0)
    target_headcount_provenance: str | None = None
    trigger_criteria: list[str] = Field(default_factory=list)
    priority_zones: list[str] = Field(default_factory=list)
    quantities: list[QuantityItem] = Field(default_factory=list)
    cited_record_ids: list[str] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)


class CoordinatorAgentOutput(_StrictModel):
    executive_summary: str = Field(min_length=1)
    action_plan: list[str] = Field(default_factory=list)
    cited_record_ids: list[str] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)


class ToolExecutionEnvelope(_StrictModel):
    call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    agent: AgentName
    status: Literal["ok", "no_data", "error", "rejected"]
    arguments: dict[str, Any] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    latency_ms: float = 0.0


class AbstentionItem(_StrictModel):
    agent: AgentName
    reason: str = Field(min_length=1)


class ConflictItem(_StrictModel):
    kind: str = Field(min_length=1)
    description: str = Field(min_length=1)
    risk_level: str
    evacuation_recommendation: str


class HumanReviewInfo(_StrictModel):
    required: bool
    reasons: list[str] = Field(default_factory=list)


class RiskSummary(_StrictModel):
    level: Literal["low", "moderate", "high", "extreme", "insufficient_data"]
    premise_valid: bool
    premise_issue: str | None = None
    summary: str
    verified_by_agent: str | None = None


class EvacuationSummary(_StrictModel):
    recommendation: Literal[
        "no_evacuation",
        "standby_advisory",
        "voluntary_evacuation",
        "mandatory_evacuation",
        "insufficient_data",
    ]
    requires_human_approval: bool = True
    target_headcount: int | None = None
    target_headcount_provenance: str | None = None
    trigger_criteria: list[str] = Field(default_factory=list)
    priority_zones: list[str] = Field(default_factory=list)
    verified_by_agent: str | None = None


class LogisticsSummary(_StrictModel):
    resource_adequacy: Literal["adequate", "constrained", "insufficient_inventory_data", "insufficient_data"]
    capacity_comparisons: list[CapacityComparison] = Field(default_factory=list)
    supply_recommendations: list[str] = Field(default_factory=list)
    verified_by_agent: str | None = None


class Briefing(_StrictModel):
    """Canonical structured Task 4 Briefing schema exported to docs/briefing_schema.json."""

    run_id: str
    status: Literal["complete", "partial", "insufficient_data"]
    risk: RiskSummary
    evacuation_recommendation: EvacuationSummary
    logistics_summary: LogisticsSummary | None = None
    quantities: list[QuantityItem] = Field(default_factory=list)
    citations: list[CitationItem] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)
    abstentions: list[AbstentionItem] = Field(default_factory=list)
    conflicts: list[ConflictItem] = Field(default_factory=list)
    human_review: HumanReviewInfo
    disclaimers: list[str] = Field(default_factory=list)
    label: Literal["historical_precedent", "hypothetical_scenario"]
    model_ids: list[str] = Field(default_factory=list)
    dataset_version: str
    executive_summary: str = ""
    action_plan: list[str] = Field(default_factory=list)


class VerifiedAgentWrite(_StrictModel):
    agent: AgentName
    writer: AgentName
    verified_at: str
    output: dict[str, Any]


class SSEEventRecord(_StrictModel):
    event: SSEEventType
    run_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    agent: AgentName | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class TokenUsageSummary(_StrictModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0


class AgentStepTrace(_StrictModel):
    step: int
    request_messages: list[dict[str, Any]]
    tools_sent: list[str]
    model: str | None = None
    temperature: float
    seed: int
    finish_reason: str | None = None
    tool_calls_received: list[dict[str, Any]] = Field(default_factory=list)
    assistant_text: str | None = None
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    verification_passed: bool | None = None
    verification_errors: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    usage: TokenUsageSummary = Field(default_factory=TokenUsageSummary)


class AgentTraceRecord(_StrictModel):
    agent: AgentName
    status: Literal["verified", "abstained", "failed"]
    steps: list[AgentStepTrace] = Field(default_factory=list)
    called_tools: list[str] = Field(default_factory=list)
    rejection_reasons: list[str] = Field(default_factory=list)
    total_latency_ms: float = 0.0
    usage: TokenUsageSummary = Field(default_factory=TokenUsageSummary)


class OrchestrationRunResult(_StrictModel):
    briefing: Briefing
    agent_traces: list[AgentTraceRecord] = Field(default_factory=list)
    events: list[SSEEventRecord] = Field(default_factory=list)
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    token_usage: TokenUsageSummary = Field(default_factory=TokenUsageSummary)
    wall_time_ms: float = 0.0


class SharedState(_StrictModel):
    """Per-run typed SharedState mutated exclusively through write_verified and tool recorders."""

    run_id: str
    dataset_version: str
    scenario_inputs: dict[str, Any] = Field(default_factory=dict)
    resolved_geography: dict[str, Any] = Field(default_factory=dict)
    user_supplied_numbers: set[str] = Field(default_factory=set)
    tool_calls: dict[str, ToolExecutionEnvelope] = Field(default_factory=dict)
    tool_history: list[ToolExecutionEnvelope] = Field(default_factory=list)
    retrieved_records: dict[str, CitationItem] = Field(default_factory=dict)
    verified_reports: dict[str, VerifiedAgentWrite] = Field(default_factory=dict)
    abstentions: list[AbstentionItem] = Field(default_factory=list)
    abstained_agents: set[str] = Field(default_factory=set)
    budget_exhausted_agents: set[str] = Field(default_factory=set)
    delegation_attempts: dict[str, int] = Field(default_factory=dict)
    run_total_tokens: int = 0
    conflicts: list[ConflictItem] = Field(default_factory=list)
    model_ids: list[str] = Field(default_factory=list)

    def record_tool_execution(self, envelope: ToolExecutionEnvelope) -> None:
        self.tool_calls[envelope.call_id] = envelope
        self.tool_history.append(envelope)
        if envelope.tool_name == "search_precedents" and envelope.status == "ok":
            for raw in envelope.data.get("matches", []):
                cit = CitationItem.model_validate(raw)
                self.retrieved_records[cit.record_id] = cit
        elif envelope.tool_name == "geocode_region" and envelope.status == "ok":
            self.resolved_geography = {
                "latitude": envelope.data.get("latitude"),
                "longitude": envelope.data.get("longitude"),
                "country": envelope.data.get("country"),
                "resolved_location": envelope.data.get("resolved_location"),
                "population": envelope.data.get("population"),
                "provenance": envelope.call_id,
            }

    def write_verified(self, agent: AgentName, writer: AgentName, validated_output: BaseModel) -> VerifiedAgentWrite:
        if agent != writer:
            raise PermissionError(f"Agent '{writer}' cannot overwrite report slot owned by '{agent}'.")
        entry = VerifiedAgentWrite(
            agent=agent,
            writer=writer,
            verified_at=datetime.now(timezone.utc).isoformat(),
            output=validated_output.model_dump(),
        )
        self.verified_reports[agent] = entry
        return entry

    def get_verified_risk(self) -> RiskAgentOutput | None:
        raw = self.verified_reports.get("risk")
        return RiskAgentOutput.model_validate(raw.output) if raw else None

    def get_verified_logistics(self) -> LogisticsAgentOutput | None:
        raw = self.verified_reports.get("logistics")
        return LogisticsAgentOutput.model_validate(raw.output) if raw else None

    def get_verified_evacuation(self) -> EvacuationAgentOutput | None:
        raw = self.verified_reports.get("evacuation")
        return EvacuationAgentOutput.model_validate(raw.output) if raw else None

    def get_verified_coordinator(self) -> CoordinatorAgentOutput | None:
        raw = self.verified_reports.get("coordinator")
        return CoordinatorAgentOutput.model_validate(raw.output) if raw else None

