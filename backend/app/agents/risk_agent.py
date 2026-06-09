from __future__ import annotations

from backend.app.agents.base import AgentContext, AgentRunner, compute_confidence, evidence
from backend.app.schemas import AgentResult


class RiskAgent(AgentRunner):
    agent_name = "risk"

    def run(self, ctx: AgentContext) -> AgentResult:
        scenario = ctx.scenario
        rag = ctx.rag
        if rag.get("status") != "ok":
            return self._result(
                scenario,
                confidence=0.0,
                reasoning_summary="Risk analysis requires retrieved historical disasters.",
                evidence_items=[],
                recommendations=[],
                next_actions=["Restore Supabase retrieval."],
                status="failed",
                output={"status": "insufficient_data"},
                execution_time_ms=0,
            )

        retrieved = rag.get("retrieved_disasters", [])
        hist_deaths = [
            float(r["total_deaths"])
            for r in retrieved
            if isinstance(r.get("total_deaths"), (int, float)) and r["total_deaths"] == r["total_deaths"]
        ]
        hist_affected = [
            float(r["total_affected"])
            for r in retrieved
            if isinstance(r.get("total_affected"), (int, float)) and r["total_affected"] == r["total_affected"]
        ]
        hist_severity = 0.0
        if hist_deaths or hist_affected:
            death_norm = min(1.0, (sum(hist_deaths) / len(hist_deaths)) / 5000) if hist_deaths else 0
            affected_norm = min(1.0, (sum(hist_affected) / len(hist_affected)) / 5_000_000) if hist_affected else 0
            hist_severity = round((death_norm * 0.55 + affected_norm * 0.45), 3)

        wind_contrib = round(min(30, scenario.wind_speed / 250 * 30), 1)
        rain_contrib = round(min(25, scenario.rainfall / 500 * 25), 1)
        density_contrib = round(min(20, scenario.population_density / 20000 * 20), 1)
        elevation_contrib = round(min(15, max(0, (50 - scenario.elevation) / 50 * 15)), 1)
        humidity_contrib = round(min(10, scenario.humidity / 100 * 10), 1)
        historical_contrib = round(hist_severity * 20, 1)

        contributors = {
            "wind": wind_contrib,
            "rainfall": rain_contrib,
            "population_density": density_contrib,
            "elevation": elevation_contrib,
            "humidity": humidity_contrib,
            "historical_severity": historical_contrib,
        }
        risk_score = round(min(99, sum(contributors.values())), 1)

        risk_level = (
            "critical" if risk_score >= 75
            else "high" if risk_score >= 55
            else "medium" if risk_score >= 35
            else "low"
        )
        affected_population = round(scenario.population * min(0.9, risk_score / 100 * 0.85))

        top_sim = float(retrieved[0].get("similarity", 0)) if retrieved else 0.0
        confidence = compute_confidence(True, len(retrieved), top_sim, [])

        return self._result(
            scenario,
            confidence=confidence,
            reasoning_summary=(
                "Transparent risk score from weather, exposure, elevation, humidity, "
                f"and historical severity derived from {len(retrieved)} retrieved disasters."
            ),
            evidence_items=[
                evidence("scenario_input", "Weather and exposure", f"wind={scenario.wind_speed}, rain={scenario.rainfall}"),
                evidence(
                    "supabase_pgvector",
                    "Historical severity anchor",
                    f"Top match {retrieved[0].get('event_name')} similarity {retrieved[0].get('similarity')}",
                    record_id=str(retrieved[0].get("id")),
                ),
            ],
            recommendations=[f"Treat scenario as {risk_level} risk ({risk_score}/100)."],
            next_actions=["Send risk profile to Prediction agent."],
            status="completed",
            output={
                "risk_score": risk_score,
                "risk_level": risk_level,
                "contributors": contributors,
                "affected_population": affected_population,
                "historical_severity": hist_severity,
                "critical_zones": [
                    {"zone": "coastal_low_elevation", "reason": f"elevation={scenario.elevation}m", "priority": 1},
                    {"zone": "high_density_corridor", "reason": f"density={scenario.population_density}", "priority": 2},
                ],
            },
            execution_time_ms=0,
        )
