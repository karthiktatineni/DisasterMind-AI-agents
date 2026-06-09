from __future__ import annotations

from typing import Any

from backend.app.agents.base import AgentContext, AgentRunner, compute_confidence, evidence
from backend.app.schemas import AgentResult


def _safe_mean(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None and v == v]
    if not clean:
        return None
    return sum(clean) / len(clean)


class KnowledgeAgent(AgentRunner):
    agent_name = "knowledge"

    async def run(self, ctx: AgentContext) -> AgentResult:
        rag = ctx.rag
        if rag.get("status") != "ok":
            return self._result(
                ctx.scenario,
                confidence=0.0,
                reasoning_summary="Historical knowledge retrieval failed; no fabricated events returned.",
                evidence_items=[],
                recommendations=["Configure Supabase pgvector and upload embeddings before planning."],
                next_actions=["Resolve retrieval configuration."],
                status="failed",
                output={"status": "insufficient_data", "historical_events": [], "sources": []},
                execution_time_ms=0,
            )

        retrieved: list[dict[str, Any]] = rag.get("retrieved_disasters", [])
        sources = rag.get("sources", [])

        deaths = [r.get("total_deaths") for r in retrieved if isinstance(r.get("total_deaths"), (int, float))]
        affected = [r.get("total_affected") for r in retrieved if isinstance(r.get("total_affected"), (int, float))]
        damage = [r.get("total_damage") for r in retrieved if isinstance(r.get("total_damage"), (int, float))]

        patterns = {
            "mortality": {"mean_deaths": _safe_mean([float(d) for d in deaths if d == d])},
            "affected_population": {"mean_affected": _safe_mean([float(a) for a in affected if a == a])},
            "economic_damage": {"mean_damage_usd_thousands": _safe_mean([float(d) for d in damage if d == d])},
        }

        top_sim = float(retrieved[0].get("similarity", 0)) if retrieved else 0.0
        confidence = compute_confidence(
            has_retrieval=True,
            match_count=len(retrieved),
            top_similarity=top_sim,
            data_gaps=[],
        )

        evidence_items = [
            evidence(
                "supabase_pgvector",
                f"Historical match: {item.get('event_name')}",
                f"Similarity {item.get('similarity')} | {item.get('country')} {item.get('start_year')}",
                record_id=str(item.get("id")),
            )
            for item in retrieved[:5]
        ]

        return self._result(
            ctx.scenario,
            confidence=confidence,
            reasoning_summary=(
                f"Retrieved {len(retrieved)} comparable disasters from Supabase pgvector "
                "with traceable source IDs and similarity scores."
            ),
            evidence_items=evidence_items,
            recommendations=[
                "Use retrieved mortality and damage patterns as planning bounds.",
                "Cross-check policy guidance separately when Foundry IQ is configured.",
            ],
            next_actions=["Pass historical context to Similarity and Risk agents."],
            status="completed",
            output={
                "historical_events": retrieved,
                "sources": sources,
                "damage_patterns": patterns["economic_damage"],
                "mortality_patterns": patterns["mortality"],
                "affected_patterns": patterns["affected_population"],
            },
            execution_time_ms=0,
        )
