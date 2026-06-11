from __future__ import annotations

from backend.app.agents.base import AgentContext, AgentRunner, compute_confidence, evidence
from backend.app.schemas import AgentResult


class ValidationAgent(AgentRunner):
    agent_name = "validation"

    async def run(self, ctx: AgentContext) -> AgentResult:
        return await self._timed_run_async(ctx, self._run_impl)

    async def _run_impl(self, ctx: AgentContext) -> AgentResult:
        scenario = ctx.scenario
        prediction = ctx.outputs.get("prediction", {})
        risk = ctx.outputs.get("risk", {})
        logistics = ctx.outputs.get("logistics", {})
        evacuation = ctx.outputs.get("evacuation", {})

        if prediction.get("status") == "insufficient_data":
            return self._result(
                scenario,
                confidence=0.0,
                reasoning_summary="Cannot validate without predictions.",
                evidence_items=[],
                recommendations=[],
                next_actions=[],
                status="failed",
                output={"passed": False, "status": "insufficient_data", "issues": ["Missing predictions"]},
                execution_time_ms=0,
            )

        issues = []
        warnings = []
        evac_need = int(prediction.get("predicted_affected", 0))
        predicted_deaths = int(prediction.get("predicted_deaths", 0))

        if evac_need > scenario.population:
            issues.append("Evacuation need exceeds total population.")
        if evac_need > scenario.shelter_capacity:
            issues.append(f"Insufficient shelters: gap of {evac_need - scenario.shelter_capacity}")
        if predicted_deaths > scenario.hospital_capacity * 0.5:
            warnings.append("Predicted casualties may exceed hospital surge capacity.")
        if evac_need > scenario.population * 0.8:
            warnings.append("Evacuation feasibility constrained by population scale.")

        shortfalls = logistics.get("resource_shortfalls", [])
        if shortfalls:
            warnings.append(f"Resource shortfalls: {shortfalls}")

        passed = len(issues) == 0
        validation_status = "failed" if issues else "passed_with_warnings" if warnings else "passed"
        confidence = compute_confidence(
            ctx.rag.get("status") == "ok",
            len(ctx.rag.get("retrieved_disasters", [])),
            float(ctx.rag.get("retrieved_disasters", [{}])[0].get("similarity", 0))
            if ctx.rag.get("retrieved_disasters")
            else 0,
            warnings,
        )

        return self._result(
            scenario,
            confidence=confidence,
            reasoning_summary="Validated shelter, hospital, evacuation feasibility against computed predictions.",
            evidence_items=[
                evidence("prediction", "Evacuation need", str(evac_need)),
                evidence("scenario_input", "Shelter capacity", str(scenario.shelter_capacity)),
                evidence("evacuation", "Shelter gap", str(evacuation.get("shelter_gap", 0))),
            ],
            recommendations=["Proceed with warnings visible" if passed else "Stop until issues resolved."],
            next_actions=["Send to Planner." if passed else "Return remediation plan."],
            status="completed" if passed else "needs_human_review",
            output={
                "passed": passed,
                "validation_status": validation_status,
                "issues": issues,
                "warnings": warnings,
                "checked": {
                    "shelter_capacity": scenario.shelter_capacity,
                    "hospital_capacity": scenario.hospital_capacity,
                    "evacuation_need": evac_need,
                    "risk_score": risk.get("risk_score"),
                },
            },
            execution_time_ms=0,
        )
