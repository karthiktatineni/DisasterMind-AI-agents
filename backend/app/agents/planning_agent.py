from __future__ import annotations

from backend.app.agents.base import AgentContext, AgentRunner, compute_confidence, evidence
from backend.app.schemas import AgentResult


import json
from backend.app.schemas import GenerationMessage
from backend.app.agents.base import STRICT_REALTIME_PROMPT
from backend.app.agents.fallbacks import build_planning_output, parse_json_object

class PlanningAgent(AgentRunner):
    agent_name = "planner"

    async def run(self, ctx: AgentContext) -> AgentResult:
        return await self._timed_run_async(ctx, self._run_impl)

    async def _run_impl(self, ctx: AgentContext) -> AgentResult:
        scenario = ctx.scenario
        risk = ctx.outputs.get("risk", {})
        prediction = ctx.outputs.get("prediction", {})
        logistics = ctx.outputs.get("logistics", {})
        evacuation = ctx.outputs.get("evacuation", {})
        rag = ctx.rag

        if rag.get("status") != "ok":
            return self._result(
                scenario,
                confidence=0.0,
                reasoning_summary="Planning requires retrieved evidence.",
                evidence_items=[],
                recommendations=[],
                next_actions=[],
                status="failed",
                output={"status": "insufficient_data"},
                execution_time_ms=0,
            )

        prompt = (
            f"Scenario: {scenario.disaster_type} in {scenario.region}.\n"
            f"Risk output: {json.dumps(risk)}\n"
            f"Prediction output: {json.dumps(prediction)}\n\n"
            f"RAG Context:\n{rag.get('context', '')}\n\n"
            "Generate a resource and evacuation plan based strictly on the retrieved context and ML predictions. "
            "Output valid JSON exactly matching this structure:\n"
            "{\n"
            '  "evacuation_strategy": "<string>",\n'
            '  "shelter_assignments": {"primary": <int>, "overflow": <int>},\n'
            '  "resource_estimates": {"ambulances": <int>, "water_units": <int>, "food_packets": <int>},\n'
            '  "phases": [{"phase": "<timeframe>", "action": "<action>", "evidence_citations": ["<citation>"]}]\n'
            "}"
        )

        try:
            if ctx.generation_client is None:
                raise RuntimeError("NVIDIA generation client is not configured.")
            generated = await ctx.generation_client.generate(
                messages=[
                    GenerationMessage(role="system", content=f"{STRICT_REALTIME_PROMPT}\nYou are a JSON-only disaster planning AI."),
                    GenerationMessage(role="user", content=prompt),
                ],
                max_tokens=1000,
                temperature=0.1,
            )
            data = parse_json_object(generated.content)
        except Exception as exc:
            data = build_planning_output(scenario, risk, prediction, logistics, evacuation, rag)
            data["fallback_reason"] = str(exc)

        retrieved = rag.get("retrieved_disasters", [])
        top_sim = float(retrieved[0].get("similarity", 0)) if retrieved else 0.0
        confidence = compute_confidence(True, len(retrieved), top_sim, [])

        return self._result(
            scenario,
            confidence=confidence,
            reasoning_summary="Dynamic operational phases generated via LLM citing RAG context.",
            evidence_items=[
                evidence("llm_planning", "strategy", data.get("evacuation_strategy", ""))
            ],
            recommendations=["Execute generated phases."],
            next_actions=["Submit plan to Logistics and Decision agents."],
            status="completed",
            output=data,
            execution_time_ms=0,
        )
