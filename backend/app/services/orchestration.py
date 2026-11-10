"""Service compatibility layer bridging the 4-agent DisasterMindOrchestrator to FastAPI."""

from __future__ import annotations

from typing import Any, Callable

from backend.app.config import Settings, get_settings
from backend.app.orchestration.models import (
    AGENT_SEQUENCE,
    Briefing,
    OrchestrationRunResult,
    SSEEventRecord,
)
from backend.app.orchestration.orchestrator import DisasterMindOrchestrator as CoreOrchestrator
from backend.app.schemas import DisasterScenario, ScenarioIntakeRequest
from backend.app.services.llm_client import NativeToolLLMClient
from backend.app.services.retrieval_service import RetrievalService

__all__ = ["AGENT_SEQUENCE", "DisasterMindOrchestrator"]


class DisasterMindOrchestrator:
    """Thin FastAPI service facade delegating exclusively to the 4-agent native tool orchestrator."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        llm_client: NativeToolLLMClient | None = None,
        retrieval_service: RetrievalService | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._core = CoreOrchestrator(
            settings=self.settings,
            llm_client=llm_client,
            retrieval_service=retrieval_service,
        )

    async def run_briefing(
        self,
        scenario: DisasterScenario | ScenarioIntakeRequest | dict[str, Any],
        *,
        run_id: str | None = None,
        on_event: Callable[[SSEEventRecord], None] | None = None,
    ) -> OrchestrationRunResult:
        return await self._core.run(scenario, run_id=run_id, on_event=on_event)

    async def run(self, scenario: DisasterScenario) -> Briefing:
        result = await self.run_briefing(scenario)
        return result.briefing

    async def run_from_intake(self, intake: ScenarioIntakeRequest) -> OrchestrationRunResult:
        return await self.run_briefing(intake)

