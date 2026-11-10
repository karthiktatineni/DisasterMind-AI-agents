from __future__ import annotations

from typing import Any
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class GenerationMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class GenerationRequest(BaseModel):
    messages: list[GenerationMessage] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)


class GenerationResponse(BaseModel):
    provider: str
    model: str
    content: str
    reasoning_content: str | None = None
    request_id: str | None = None


class DisasterScenario(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    disaster_type: str = Field(default="Cyclone", min_length=1)
    region: str = Field(default="Unknown region", min_length=1)
    time_horizon: str = Field(default="48 hours", min_length=1)
    severity: Literal["Low", "Moderate", "High", "Critical"] = "High"
    population: int | None = Field(default=None, ge=0)
    hospital_capacity: int | None = Field(default=None, ge=0)
    shelter_capacity: int | None = Field(default=None, ge=0)
    wind_speed: float | None = Field(default=None, ge=0)
    rainfall: float | None = Field(default=None, ge=0)
    elevation: float | None = Field(default=None, ge=-500, le=9000)
    population_density: float | None = Field(default=None, ge=0)
    temperature: float | None = Field(default=None, ge=-80, le=80)
    humidity: float | None = Field(default=None, ge=0, le=100)
    historical_damage: float | None = Field(default=None, ge=0, le=1)
    notes: str = Field(default="", max_length=500)
    latitude: float | None = None
    longitude: float | None = None
    country: str | None = None
    resolved_location: str | None = None
    data_sources: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("disaster_type", "region", "time_horizon")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value cannot be empty")
        return stripped


class ScenarioIntakeRequest(BaseModel):
    disaster_type: str = Field(min_length=1)
    region: str = Field(min_length=1)
    severity: Literal["Low", "Moderate", "High", "Critical"] = "Moderate"
    time_horizon: str = Field(default="24 hours", min_length=1)
    notes: str = Field(default="", max_length=500)

    @field_validator("disaster_type", "region", "time_horizon")
    @classmethod
    def strip_intake_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value cannot be empty")
        return stripped


class AgentResult(BaseModel):
    request_id: str
    agent_name: str
    timestamp: str
    confidence: float = Field(ge=0, le=1)
    reasoning_summary: str
    evidence: list[dict[str, Any]]
    recommendations: list[str]
    next_actions: list[str]
    status: Literal[
        "completed",
        "partial",
        "failed",
        "needs_human_review",
        "passed",
        "passed_with_warnings",
    ]
    output: dict[str, Any]


class OrchestrationResponse(BaseModel):
    request_id: str
    status: Literal["completed", "completed_with_warnings", "failed"]
    agent_sequence: list[str]
    agent_results: list[AgentResult]
    final_report: str
    validation_status: str
    confidence: float = Field(ge=0, le=1)
    provider: str
    model: str | None = None
    warnings: list[str]
    retrieved_disasters: list[dict[str, Any]] = Field(default_factory=list)
    evidence_used: list[dict[str, Any]] = Field(default_factory=list)
    insufficient_data: dict[str, Any] | None = None


class AgentQuestionRequest(BaseModel):
    scenario: DisasterScenario | None = None
    intake: ScenarioIntakeRequest | None = None
    question: str = Field(min_length=3, max_length=500)


class AgentQuestionResponse(BaseModel):
    request_id: str
    answer: str
    provider: str
    model: str | None = None
    cited_agents: list[str]
    confidence: float = Field(ge=0, le=1)


class RetrievalMatch(BaseModel):
    id: str
    event_name: str
    display_label: str | None = None
    disaster_type: str | None = None
    disaster_subtype: str | None = None
    category_tag: str | None = None
    country: str | None = None
    region: str | None = None
    location: str | None = None
    start_year: int | None = None
    total_deaths: float | None = None
    total_affected: float | None = None
    total_damage: float | None = None
    similarity: float
    search_text: str | None = None


class RetrievalResult(BaseModel):
    status: Literal["ok", "no_data", "error"]
    record_ids: list[str] = Field(default_factory=list)
    matches: list[RetrievalMatch] = Field(default_factory=list)
    threshold_used: float
    top_score: float | None = None
    namespace: str
    dataset_version: str | None = None
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None
