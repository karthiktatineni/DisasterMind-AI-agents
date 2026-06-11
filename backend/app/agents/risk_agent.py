from __future__ import annotations

from typing import Any

from backend.app.agents.base import AgentContext, AgentRunner, compute_confidence, evidence
from backend.app.agents.fallbacks import build_risk_profile, parse_json_object
from backend.app.schemas import AgentResult

SEVERITY_WEIGHT = {"Low": 0.15, "Moderate": 0.35, "High": 0.65, "Critical": 0.9}


def _clean_number(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or value != value:
        return None
    return float(value)


def _weighted_mean(rows: list[dict[str, Any]], field: str) -> float | None:
    weighted_total = 0.0
    total_weight = 0.0
    for row in rows:
        value = _clean_number(row.get(field))
        if value is None:
            continue
        weight = max(0.01, float(row.get("similarity", 0) or 0))
        weighted_total += value * weight
        total_weight += weight
    if total_weight == 0:
        return None
    return weighted_total / total_weight


from backend.app.schemas import GenerationMessage
from backend.app.agents.base import STRICT_REALTIME_PROMPT

class RiskAgent(AgentRunner):
    agent_name = "risk"

    async def run(self, ctx: AgentContext) -> AgentResult:
        return await self._timed_run_async(ctx, self._run_impl)

    async def _run_impl(self, ctx: AgentContext) -> AgentResult:
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

        prompt = (
            f"Scenario: {scenario.disaster_type} in {scenario.region}. Severity: {scenario.severity}.\n\n"
            f"RAG Context:\n{rag.get('context', '')}\n\n"
            "Analyze the risk using only the retrieved evidence. Output valid JSON exactly matching this structure:\n"
            "{\n"
            '  "risk_score": <int 0-100>,\n'
            '  "risk_level": "<low|medium|high|critical>",\n'
            '  "contributors": {"factor_name": <score>},\n'
            '  "affected_population_estimate": <int>,\n'
            '  "critical_zones": [{"zone": "<name>", "reason": "<reason>", "priority": <int>}]\n'
            "}"
        )

        try:
            if ctx.generation_client is None:
                raise RuntimeError("NVIDIA generation client is not configured.")
            generated = await ctx.generation_client.generate(
                messages=[
                    GenerationMessage(role="system", content=f"{STRICT_REALTIME_PROMPT}\nYou are a strict JSON-only AI risk analyst."),
                    GenerationMessage(role="user", content=prompt),
                ],
                max_tokens=800,
                temperature=0.1,
            )
            data = parse_json_object(generated.content)
            risk_score = data.get("risk_score", 0)
        except Exception as exc:
            data = build_risk_profile(scenario, rag)
            data["fallback_reason"] = str(exc)
            risk_score = data.get("risk_score", 0)

        retrieved = rag.get("retrieved_disasters", [])
        top_sim = float(retrieved[0].get("similarity", 0)) if retrieved else 0.0
        confidence = compute_confidence(True, len(retrieved), top_sim, [])

        evidence_items = []
        if retrieved:
            evidence_items.append(evidence(
                "supabase_pgvector",
                "Top Match Anchor",
                f"Top match {retrieved[0].get('event_name')} similarity {retrieved[0].get('similarity')}",
                record_id=str(retrieved[0].get("id")),
            ))
            
        return self._result(
            scenario,
            confidence=confidence,
            reasoning_summary="Dynamic risk score derived from LLM analysis of retrieved historical matches.",
            evidence_items=evidence_items,
            recommendations=[f"Treat scenario as {data.get('risk_level', 'unknown')} risk ({risk_score}/100)."],
            next_actions=["Send risk profile to Prediction agent."],
            status="completed",
            output=data,
            execution_time_ms=0,
        )
