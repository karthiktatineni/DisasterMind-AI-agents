from __future__ import annotations

from backend.app.agents.base import AgentContext, AgentRunner, compute_confidence, evidence
from backend.app.schemas import AgentResult, DisasterScenario


def _metadata_boost(scenario: DisasterScenario, match: dict) -> float:
    boost = 0.0
    if scenario.country and match.get("country"):
        if scenario.country.lower() in str(match["country"]).lower():
            boost += 0.03
    if scenario.disaster_type and match.get("disaster_type"):
        if scenario.disaster_type.lower() in str(match["disaster_type"]).lower():
            boost += 0.04
    return boost


class SimilarityAgent(AgentRunner):
    agent_name = "similarity"

    def run(self, ctx: AgentContext) -> AgentResult:
        similarity = ctx.similarity
        if similarity.get("status") != "ok":
            return self._result(
                ctx.scenario,
                confidence=0.0,
                reasoning_summary="Vector similarity search unavailable.",
                evidence_items=[],
                recommendations=[],
                next_actions=["Fix Supabase retrieval."],
                status="failed",
                output={"status": "insufficient_data", "top_matches": [], "reasoning": []},
                execution_time_ms=0,
            )

        scenario = ctx.scenario
        raw_matches = similarity.get("matches", [])
        ranked = []
        reasoning = []

        for match in raw_matches:
            vector_sim = float(match.get("similarity", 0))
            meta_boost = _metadata_boost(scenario, match)
            combined = round(min(0.999, vector_sim + meta_boost), 4)
            ranked.append({**match, "vector_similarity": vector_sim, "combined_score": combined})
            reasoning.append(
                f"{match.get('event_name')}: pgvector={vector_sim:.3f}, "
                f"metadata_boost={meta_boost:.3f}, combined={combined:.3f}"
            )

        ranked.sort(key=lambda row: row["combined_score"], reverse=True)
        top_sim = ranked[0]["combined_score"] if ranked else 0.0
        confidence = compute_confidence(
            has_retrieval=True,
            match_count=len(ranked),
            top_similarity=top_sim,
            data_gaps=[],
        )

        return self._result(
            scenario,
            confidence=confidence,
            reasoning_summary="Ranked nearest historical disasters using pgvector similarity with location and type weighting.",
            evidence_items=[
                evidence(
                    "supabase_pgvector",
                    str(row.get("event_name")),
                    f"Similarity {row.get('combined_score')}",
                    record_id=str(row.get("id")),
                )
                for row in ranked[:5]
            ],
            recommendations=["Use top 3 matches as primary evidence for predictions and planning."],
            next_actions=["Send ranked matches to Risk and Prediction agents."],
            status="completed",
            output={"top_matches": ranked, "reasoning": reasoning},
            execution_time_ms=float(similarity.get("execution_time_ms", 0)),
        )
