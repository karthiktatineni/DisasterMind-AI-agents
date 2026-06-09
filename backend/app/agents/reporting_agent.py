from __future__ import annotations

from backend.app.agents.base import AgentContext, AgentRunner, compute_confidence, evidence
from backend.app.schemas import AgentResult


class ReportingAgent(AgentRunner):
    agent_name = "reporting"

    async def run(self, ctx: AgentContext, report_text: str = "", provider: str = "") -> AgentResult:
        decision = ctx.outputs.get("decision", {})
        validation = ctx.outputs.get("validation", {})
        planner = ctx.outputs.get("planner", {})
        prediction = ctx.outputs.get("prediction", {})
        retrieved = ctx.rag.get("retrieved_disasters", [])

        if not retrieved or decision.get("status") == "insufficient_data":
            return self._result(
                ctx.scenario,
                confidence=0.0,
                reasoning_summary="Reporting requires decision output with retrieved evidence.",
                evidence_items=[],
                recommendations=[],
                next_actions=[],
                status="failed",
                output={"status": "insufficient_data"},
                execution_time_ms=0,
            )

        citation_block = "\n".join(
            f"- {r.get('event_name')} ({r.get('start_year')}) | {r.get('country')} | "
            f"Similarity {round(float(r.get('similarity', 0)) * 100)}% | ID: {r.get('id')}"
            for r in retrieved[:5]
        )

        sections = {
            "executive_summary": decision.get("executive_recommendation", ""),
            "emergency_action_plan": planner.get("phases", []),
            "resource_deployment_plan": planner.get("resource_estimates", {}),
            "evacuation_plan": planner.get("evacuation_strategy", ""),
            "simulation_findings": ctx.outputs.get("simulation", {}).get("scenario_results", []),
            "evidence_used": citation_block,
            "predictions": {
                "affected": prediction.get("predicted_affected"),
                "deaths": prediction.get("predicted_deaths"),
                "damage_usd": prediction.get("predicted_damage"),
            },
        }

        if not report_text:
            report_text = (
                "Executive Summary\n"
                f"{sections['executive_summary']}\n\n"
                "Recommendation based on:\n"
                f"{citation_block}\n\n"
                "Predictions\n"
                f"Affected: {prediction.get('predicted_affected')} | "
                f"Deaths: {prediction.get('predicted_deaths')} | "
                f"Damage USD: {prediction.get('predicted_damage')}\n\n"
                "Validation\n"
                f"Status: {validation.get('validation_status')} | Issues: {validation.get('issues', [])}"
            )

        top_sim = float(retrieved[0].get("similarity", 0))
        confidence = compute_confidence(True, len(retrieved), top_sim, validation.get("warnings", []))

        return self._result(
            ctx.scenario,
            confidence=confidence,
            reasoning_summary="Generated evidence-backed report sections with disaster citations and model outputs.",
            evidence_items=[
                evidence("supabase_pgvector", "Evidence used", citation_block),
            ],
            recommendations=["Export report with evidence appendix for judge verification."],
            next_actions=["Deliver to dashboard."],
            status="completed",
            output={
                "sections": sections,
                "report_text": report_text,
                "generation_provider": provider,
                "evidence_used": [
                    {
                        "id": r.get("id"),
                        "event_name": r.get("event_name"),
                        "similarity": r.get("similarity"),
                        "country": r.get("country"),
                        "year": r.get("start_year"),
                    }
                    for r in retrieved
                ],
            },
            execution_time_ms=0,
        )
