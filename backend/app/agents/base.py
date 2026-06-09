from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any, Literal

from backend.app.schemas import AgentResult, DisasterScenario

AgentStatus = Literal[
    "completed",
    "partial",
    "failed",
    "needs_human_review",
    "passed",
    "passed_with_warnings",
]

STRICT_REALTIME_PROMPT = """STRICT REAL-TIME DATA ONLY AI PROMPT
You are an AI assistant that MUST use only real-time, up-to-date information from live sources.
Rules you must follow without exception:
Do NOT use training data, memory, assumptions, or general knowledge.
Do NOT guess or estimate any values.
If real-time data is not available, clearly say: "Real-time data not available for this query."
Always prioritize the latest available information from verified live sources.
If multiple sources conflict, prefer the most recent timestamp.
Always include timestamps or "last updated" info when available.
Never fabricate numbers, statistics, prices, news, or events.
If asked for current events, always verify using live search before responding.
If the system cannot access live data, refuse to answer rather than guessing.
Output format rules:
Provide only factual, current data.
No opinions unless explicitly requested.
No placeholders or generic responses.
Keep answers concise and data-focused.
You MUST output valid JSON only."""

def timestamp() -> str:
    return datetime.now(UTC).isoformat()


def evidence(source: str, title: str, detail: str, record_id: str | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {"source": source, "title": title, "detail": detail}
    if record_id:
        item["record_id"] = record_id
    return item


def compute_confidence(
    has_retrieval: bool,
    match_count: int,
    top_similarity: float,
    data_gaps: list[str],
) -> float:
    if not has_retrieval or match_count == 0:
        return 0.0
    base = min(0.95, 0.35 + top_similarity * 0.45 + min(match_count, 10) * 0.02)
    penalty = min(0.35, len(data_gaps) * 0.08)
    return round(max(0.05, base - penalty), 3)


class AgentContext:
    def __init__(self, scenario: DisasterScenario, generation_client=None):
        self.scenario = scenario
        self.outputs: dict[str, dict[str, Any]] = {}
        self.retrieval: dict[str, Any] = {}
        self.similarity: dict[str, Any] = {}
        self.rag: dict[str, Any] = {}
        self.generation_client = generation_client

    def store(self, agent_name: str, output: dict[str, Any]) -> None:
        self.outputs[agent_name] = output

class AgentRunner:
    agent_name: str = "base"

    async def run(self, ctx: AgentContext) -> AgentResult:
        raise NotImplementedError

    def _result(
        self,
        scenario: DisasterScenario,
        *,
        confidence: float,
        reasoning_summary: str,
        evidence_items: list[dict[str, Any]],
        recommendations: list[str],
        next_actions: list[str],
        status: AgentStatus,
        output: dict[str, Any],
        execution_time_ms: float,
    ) -> AgentResult:
        output = {**output, "execution_time_ms": round(execution_time_ms, 2)}
        return AgentResult(
            request_id=scenario.request_id,
            agent_name=self.agent_name,
            timestamp=timestamp(),
            confidence=max(0.0, min(1.0, confidence)),
            reasoning_summary=reasoning_summary,
            evidence=evidence_items,
            recommendations=recommendations,
            next_actions=next_actions,
            status=status,
            output=output,
        )

    def _timed_run(self, scenario: DisasterScenario, fn) -> AgentResult:
        started = time.perf_counter()
        result = fn()
        elapsed = (time.perf_counter() - started) * 1000
        result.output["execution_time_ms"] = round(elapsed, 2)
        return result
