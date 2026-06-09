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
    region: str = Field(default="Visakhapatnam coastal zone", min_length=1)
    time_horizon: str = Field(default="48 hours", min_length=1)
    severity: Literal["Low", "Moderate", "High", "Critical"] = "Critical"
    population: int = Field(default=186000, ge=0)
    hospital_capacity: int = Field(default=4200, ge=0)
    shelter_capacity: int = Field(default=132000, ge=0)
    wind_speed: float = Field(default=132, ge=0)
    rainfall: float = Field(default=238, ge=0)
    elevation: float = Field(default=8, ge=-500, le=9000)
    population_density: float = Field(default=6100, ge=0)
    temperature: float = Field(default=30, ge=-80, le=80)
    humidity: float = Field(default=82, ge=0, le=100)
    historical_damage: float = Field(default=0.55, ge=0, le=1)
    notes: str = Field(default="", max_length=4000)
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
    notes: str = Field(default="", max_length=4000)

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
    question: str = Field(min_length=3, max_length=2000)


class AgentQuestionResponse(BaseModel):
    request_id: str
    answer: str
    provider: str
    model: str | None = None
    cited_agents: list[str]
    confidence: float = Field(ge=0, le=1)
