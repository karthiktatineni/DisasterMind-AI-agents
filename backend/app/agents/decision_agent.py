import json
from backend.app.agents.base import AgentContext, AgentRunner, compute_confidence, evidence, STRICT_REALTIME_PROMPT
from backend.app.agents.fallbacks import build_decision_output, parse_json_object
from backend.app.schemas import AgentResult, GenerationMessage

class DecisionAgent(AgentRunner):
    agent_name = "decision"

    async def run(self, ctx: AgentContext) -> AgentResult:
        return await self._timed_run_async(ctx, self._run_impl)

    async def _run_impl(self, ctx: AgentContext) -> AgentResult:
        validation = ctx.outputs.get("validation", {})
        planner = ctx.outputs.get("planner", {})
        prediction = ctx.outputs.get("prediction", {})
        rag = ctx.rag

        if validation.get("status") == "insufficient_data" or rag.get("status") != "ok":
            return self._result(
                ctx.scenario,
                confidence=0.0,
                reasoning_summary="Decision synthesis requires validation and retrieved disaster evidence.",
                evidence_items=[],
                recommendations=[],
                next_actions=[],
                status="failed",
                output={"status": "insufficient_data"},
                execution_time_ms=0,
            )

        failed = validation.get("validation_status") == "failed"

        prompt = (
            f"Scenario: {ctx.scenario.disaster_type} in {ctx.scenario.region}.\n"
            f"Validation Output: {json.dumps(validation)}\n"
            f"Planner Output: {json.dumps(planner)}\n"
            f"Prediction Output: {json.dumps(prediction)}\n\n"
            f"RAG Context:\n{rag.get('context', '')}\n\n"
            "Synthesize an executive decision recommendation based strictly on the provided real-time data, ML predictions, and retrieved evidence. "
            "Output valid JSON exactly matching this structure:\n"
            "{\n"
            '  "executive_recommendation": "<string>",\n'
            '  "prioritized_actions": [{"phase": "<string>", "action": "<string>"}],\n'
            '  "unresolved_risks": ["<string>"]\n'
            "}"
        )

        try:
            if ctx.generation_client is None:
                raise RuntimeError("NVIDIA generation client is not configured.")
            generated = await ctx.generation_client.generate(
                messages=[
                    GenerationMessage(role="system", content=f"{STRICT_REALTIME_PROMPT}\nYou are a strict JSON-only AI disaster executive decision maker."),
                    GenerationMessage(role="user", content=prompt),
                ],
                max_tokens=800,
                temperature=0.1,
            )
            data = parse_json_object(generated.content)
        except Exception as exc:
            data = build_decision_output(ctx.scenario, validation, planner, prediction, rag)
            data["fallback_reason"] = str(exc)

        retrieved = rag.get("retrieved_disasters", [])
        citations = [
            f"{r.get('event_name')} ({r.get('start_year')}) similarity {round(float(r.get('similarity', 0)) * 100)}%"
            for r in retrieved[:5]
        ]

        confidence = compute_confidence(
            True,
            len(retrieved),
            float(retrieved[0].get("similarity", 0)) if retrieved else 0.0,
            validation.get("warnings", []),
        )

        return self._result(
            ctx.scenario,
            confidence=confidence,
            reasoning_summary="Merged predictions, validation, and planning using LLM synthesis with explicit disaster citations.",
            evidence_items=[
                evidence("supabase_pgvector", "Retrieved disasters", "; ".join(citations)),
                evidence("validation", "Validation Status", str(validation.get("validation_status"))),
            ],
            recommendations=[data.get("executive_recommendation", "")],
            next_actions=["Generate executive report."],
            status="needs_human_review" if failed else "completed",
            output={
                "executive_recommendation": data.get("executive_recommendation", ""),
                "evidence_citations": citations,
                "prioritized_actions": data.get("prioritized_actions", []),
                "model_outputs": prediction,
                "validation_results": validation,
                "unresolved_risks": data.get("unresolved_risks", []),
                "generation_mode": data.get("generation_mode", "nvidia-build"),
                "fallback_reason": data.get("fallback_reason"),
            },
            execution_time_ms=0,
        )
