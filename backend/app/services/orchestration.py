from __future__ import annotations

from typing import Any

from backend.app.agents.decision_agent import DecisionAgent
from backend.app.agents.explainability_agent import ExplainabilityAgent
from backend.app.agents.knowledge_agent import KnowledgeAgent
from backend.app.agents.operational_agents import (
    AGENT_SEQUENCE,
    CoordinatorAgent,
    EvacuationAgent,
    LogisticsAgent,
    SimulationAgent,
)
from backend.app.agents.planning_agent import PlanningAgent
from backend.app.agents.prediction_agent import PredictionAgent
from backend.app.agents.reporting_agent import ReportingAgent
from backend.app.agents.risk_agent import RiskAgent
from backend.app.agents.similarity_agent import SimilarityAgent
from backend.app.agents.validation_agent import ValidationAgent
from backend.app.agents.base import AgentContext, STRICT_REALTIME_PROMPT
from backend.app.schemas import (
    AgentQuestionResponse,
    AgentResult,
    DisasterScenario,
    GenerationMessage,
    OrchestrationResponse,
    ScenarioIntakeRequest,
)
from backend.app.services.data_enrichment import enrich_scenario
from backend.app.services.nvidia_generation import (
    NvidiaBuildGenerationClient,
    NvidiaGenerationConfigError,
    NvidiaGenerationError,
)
from backend.app.services.rag_service import RAGService
from backend.app.services.similarity_service import SimilarityService

__all__ = ["AGENT_SEQUENCE", "DisasterMindOrchestrator"]


class DisasterMindOrchestrator:
    def __init__(
        self,
        generation_client: NvidiaBuildGenerationClient | None = None,
        similarity_service: SimilarityService | None = None,
        rag_service: RAGService | None = None,
    ):
        if generation_client is not None:
            self.generation_client = generation_client
        else:
            try:
                self.generation_client = NvidiaBuildGenerationClient()
            except NvidiaGenerationConfigError:
                self.generation_client = None
        self.similarity_service = similarity_service or SimilarityService()
        self.rag_service = rag_service or RAGService()

    async def run(self, scenario: DisasterScenario) -> OrchestrationResponse:
        ctx = AgentContext(scenario, generation_client=self.generation_client)

        ctx.similarity = await self.similarity_service.search_scenario(scenario, top_k=10)
        ctx.rag = await self.rag_service.retrieve(scenario, top_k=10)

        if ctx.rag.get("status") != "ok":
            return self._insufficient_data_response(scenario, ctx.rag.get("reason", "Retrieval failed"))

        runners = [
            CoordinatorAgent(),
            KnowledgeAgent(),
            SimilarityAgent(),
            RiskAgent(),
            PredictionAgent(),
            ExplainabilityAgent(),
            LogisticsAgent(),
            EvacuationAgent(),
            SimulationAgent(),
            ValidationAgent(),
            PlanningAgent(),
            DecisionAgent(),
        ]

        results: list[AgentResult] = []
        for runner in runners:
            result = await runner.run(ctx)
            ctx.store(runner.agent_name, result.output)
            results.append(result)

        validation = results[9]
        decision = results[11]
        report_text, provider, model = await self._run_reporting_generation(scenario, results, validation, decision)

        reporting = await ReportingAgent().run(ctx, report_text=report_text, provider=provider)
        results.append(reporting)

        warnings = list(validation.output.get("warnings", []))
        blocking = validation.output.get("issues", [])
        status = (
            "failed"
            if validation.output.get("validation_status") == "failed" or blocking
            else "completed_with_warnings"
            if warnings
            else "completed"
        )
        confidence = round(sum(r.confidence for r in results) / len(results), 3) if results else 0.0

        return OrchestrationResponse(
            request_id=scenario.request_id,
            status=status,
            agent_sequence=AGENT_SEQUENCE,
            agent_results=results,
            final_report=reporting.output.get("report_text", report_text),
            validation_status=validation.output.get("validation_status", "failed"),
            confidence=confidence,
            provider=provider,
            model=model,
            warnings=warnings,
            retrieved_disasters=ctx.rag.get("retrieved_disasters", []),
            evidence_used=reporting.output.get("evidence_used", []),
        )

    def _insufficient_data_response(self, scenario: DisasterScenario, reason: str) -> OrchestrationResponse:
        return OrchestrationResponse(
            request_id=scenario.request_id,
            status="failed",
            agent_sequence=AGENT_SEQUENCE,
            agent_results=[],
            final_report="",
            validation_status="failed",
            confidence=0.0,
            provider="none",
            model=None,
            warnings=[reason],
            retrieved_disasters=[],
            evidence_used=[],
            insufficient_data={"status": "insufficient_data", "reason": reason},
        )

    async def run_from_intake(self, intake: ScenarioIntakeRequest) -> OrchestrationResponse:
        scenario = await enrich_scenario(intake)
        return await self.run(scenario)

    async def answer_question(
        self,
        scenario: DisasterScenario,
        question: str,
    ) -> AgentQuestionResponse:
        orchestration = await self.run(scenario)
        if orchestration.insufficient_data:
            return AgentQuestionResponse(
                request_id=orchestration.request_id,
                answer='{"status": "insufficient_data"}',
                provider="none",
                model=None,
                cited_agents=[],
                confidence=0.0,
            )

        cited_agents = ["risk", "prediction", "logistics", "evacuation", "validation", "decision"]
        context = {
            result.agent_name: {
                "confidence": result.confidence,
                "status": result.status,
                "recommendations": result.recommendations,
                "output": result.output,
            }
            for result in orchestration.agent_results
            if result.agent_name in cited_agents
        }

        try:
            client = self.generation_client or NvidiaBuildGenerationClient()
            generated = await client.generate(
                messages=[
                    GenerationMessage(
                        role="system",
                        content=(
                            f"{STRICT_REALTIME_PROMPT}\n"
                            "Answer only from structured agent outputs and retrieved disaster evidence. "
                            "Never invent disasters or similarity scores."
                        ),
                    ),
                    GenerationMessage(
                        role="user",
                        content=(
                            f"Question: {question}\n\n"
                            f"Retrieved disasters: {orchestration.retrieved_disasters}\n\n"
                            f"Agent outputs: {context}"
                        ),
                    ),
                ],
                max_tokens=700,
                temperature=0.15,
            )
            return AgentQuestionResponse(
                request_id=orchestration.request_id,
                answer=generated.content,
                provider=generated.provider,
                model=generated.model,
                cited_agents=cited_agents,
                confidence=orchestration.confidence,
            )
        except (NvidiaGenerationConfigError, NvidiaGenerationError):
            return AgentQuestionResponse(
                request_id=orchestration.request_id,
                answer=self._fallback_answer(question, orchestration),
                provider="local-fallback",
                model=None,
                cited_agents=cited_agents,
                confidence=orchestration.confidence,
            )

    async def answer_question_from_intake(
        self,
        intake: ScenarioIntakeRequest,
        question: str,
    ) -> AgentQuestionResponse:
        scenario = await enrich_scenario(intake)
        return await self.answer_question(scenario, question)

    async def _run_reporting_generation(
        self,
        scenario: DisasterScenario,
        results: list[AgentResult],
        validation: AgentResult,
        decision: AgentResult,
    ) -> tuple[str, str, str | None]:
        retrieved = [
            item
            for result in results
            if result.agent_name == "knowledge"
            for item in result.output.get("historical_events", [])
        ]
        citation_block = "\n".join(
            f"- {r.get('event_name')} ({r.get('start_year')}) similarity {round(float(r.get('similarity', 0)) * 100)}%"
            for r in retrieved[:5]
        )
        prompt = (
            f"Scenario: {scenario.disaster_type} in {scenario.region}\n"
            f"Evidence:\n{citation_block}\n"
            f"Decision: {decision.output}\n"
            f"Validation: {validation.output}\n"
            "Write executive summary citing only the evidence above."
        )
        try:
            client = self.generation_client or NvidiaBuildGenerationClient()
            generated = await client.generate(
                messages=[
                    GenerationMessage(
                        role="system",
                        content=f"{STRICT_REALTIME_PROMPT}\nDisasterMind Reporting Agent. Cite only provided disasters.",
                    ),
                    GenerationMessage(role="user", content=prompt),
                ],
                max_tokens=900,
                temperature=0.2,
            )
            return generated.content, generated.provider, generated.model
        except (NvidiaGenerationConfigError, NvidiaGenerationError):
            return "", "deterministic", None

    def _fallback_answer(self, question: str, orchestration: OrchestrationResponse) -> str:
        outputs = {result.agent_name: result.output for result in orchestration.agent_results}
        return (
            f"Answer to: {question}\n"
            f"Risk: {outputs.get('risk', {})}\n"
            f"Prediction: {outputs.get('prediction', {})}\n"
            f"Evidence: {orchestration.evidence_used}"
        )
