"""Task 4 verification runner: executes live 4-agent run + scripted rejection/precondition/termination/conflict proofs."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import get_settings
from backend.app.orchestration.models import AGENT_SEQUENCE, Briefing
from backend.app.orchestration.orchestrator import AGENT_SPECS, DisasterMindOrchestrator
from backend.app.schemas import DisasterScenario, ScenarioIntakeRequest
from backend.app.services.historical_stats import HistoricalStatsService
from backend.app.services.llm_client import LLMToolResponse, ToolCallRequest
from tests.test_task4_orchestration import (
    FakeRetrievalService,
    ScriptedToolLLMClient,
    _happy_handler,
    _make_fake_http,
    _make_fake_stats,
    _tc,
)

OUT_DIR = PROJECT_ROOT / "verification" / "task_4"


async def capture_scripted_gates() -> dict[str, Any]:
    """Capture explicit scripted gate traces: rejections, precondition, bounded termination, conflict."""
    settings = get_settings()

    # 1. Precondition + Conflict + All 4 Scripted Rejection Modes
    risk_step = {"n": 0}
    coord_step = {"n": 0}

    def rej_conf_handler(*, messages: list[dict[str, Any]], tools: list[dict[str, Any]], idx: int) -> LLMToolResponse:
        names = {t["function"]["name"] for t in tools}
        if "submit_coordinator_report" in names:
            coord_step["n"] += 1
            s = coord_step["n"]
            if s == 1:
                tc = _tc("c1", "run_evacuation_agent", {"focus_question": "Evacuate before risk"})
            elif s == 2:
                tc = _tc("c2", "run_risk_agent", {"focus_question": "Assess storm risk first"})
            elif s == 3:
                tc = _tc("c3", "run_logistics_agent", {"focus_question": "Assess Sphere standards"})
            elif s == 4:
                tc = _tc("c4", "run_evacuation_agent", {"focus_question": "Assess evacuation"})
            else:
                tc = _tc(
                    "c5",
                    "submit_coordinator_report",
                    {
                        "executive_summary": "Conflict case completed with flagged Risk/Evacuation disagreement.",
                        "action_plan": ["Require commander adjudication"],
                        "cited_record_ids": ["2019-0123-IND"],
                        "data_gaps": [],
                    },
                )
            return LLMToolResponse(model="scripted-gate", request_id=f"c-{s}", finish_reason="tool_calls", content=None, reasoning_content=None, tool_calls=[tc])

        if "submit_risk_report" in names:
            risk_step["n"] += 1
            s = risk_step["n"]
            if s == 1:
                tc = _tc("r1", "unknown_satellite_tool", {"x": 1})
            elif s == 2:
                tc = _tc("r2", "compute_logistics_requirements", {"population": 5000, "population_provenance": "user"})
            elif s == 3:
                return LLMToolResponse(
                    model="scripted-gate",
                    request_id="r-3",
                    finish_reason="tool_calls",
                    content=None,
                    reasoning_content=None,
                    tool_calls=[ToolCallRequest(id="r3", name="search_precedents", raw_arguments="{invalid_json")],
                )
            elif s == 4:
                tc = _tc(
                    "r4",
                    "submit_risk_report",
                    {"risk_level": "low", "premise_valid": True, "hazard_summary": "Skipped required tool attempt.", "cited_record_ids": [], "quantities": [], "data_gaps": []},
                )
            elif s == 5:
                tc = _tc("r5", "search_precedents", {"query": "Odisha storm", "disaster_type": "Storm"})
            else:
                tc = _tc(
                    "r6",
                    "submit_risk_report",
                    {"risk_level": "low", "premise_valid": True, "hazard_summary": "Low risk score assessed.", "cited_record_ids": ["2019-0123-IND"], "quantities": [], "data_gaps": []},
                )
            return LLMToolResponse(model="scripted-gate", request_id=f"r-{s}", finish_reason="tool_calls", content=None, reasoning_content=None, tool_calls=[tc])

        return _happy_handler(risk_level="low", evac_rec="mandatory_evacuation")(messages=messages, tools=tools, idx=idx)

    async with _make_fake_http() as http_client:
        orch = DisasterMindOrchestrator(
            settings=settings,
            llm_client=ScriptedToolLLMClient(rej_conf_handler),  # type: ignore[arg-type]
            retrieval_service=FakeRetrievalService(),  # type: ignore[arg-type]
            historical_stats_service=_make_fake_stats(),
            http_client=http_client,
        )
        rej_conf_res = await orch.run({"disaster_type": "Storm", "region": "Odisha"}, run_id="gate-rej-conflict")

        # 2. Bounded termination -> partial briefing
        bounded_specs = {
            k: (type(v)(**{**v.__dict__, "max_steps": 2}) if k == "coordinator" else v)
            for k, v in AGENT_SPECS.items()
        }

        def loop_coord(*, messages: list[dict[str, Any]], tools: list[dict[str, Any]], idx: int) -> LLMToolResponse:
            names = {t["function"]["name"] for t in tools}
            if "submit_coordinator_report" in names:
                tmsgs = [m for m in messages if m.get("role") == "tool"]
                if not tmsgs:
                    return LLMToolResponse(model="scripted-gate", request_id="b1", finish_reason="tool_calls", content=None, reasoning_content=None, tool_calls=[_tc("b1", "run_risk_agent", {"focus_question": "Risk check"})])
                return LLMToolResponse(model="scripted-gate", request_id="b2", finish_reason="stop", content="Free text without tool call", reasoning_content=None, tool_calls=[])
            return _happy_handler()(messages=messages, tools=tools, idx=idx)

        orch_b = DisasterMindOrchestrator(
            settings=settings,
            llm_client=ScriptedToolLLMClient(loop_coord),  # type: ignore[arg-type]
            retrieval_service=FakeRetrievalService(),  # type: ignore[arg-type]
            historical_stats_service=_make_fake_stats(),
            http_client=http_client,
            agent_specs=bounded_specs,
        )
        bounded_res = await orch_b.run({"disaster_type": "Storm", "region": "Odisha"}, run_id="gate-bounded")

    return {
        "rejection_precondition_conflict_case": {
            "briefing_status": rej_conf_res.briefing.status,
            "conflicts": [c.model_dump() for c in rej_conf_res.briefing.conflicts],
            "human_review": rej_conf_res.briefing.human_review.model_dump(),
            "coordinator_precondition_result": rej_conf_res.agent_traces[0].steps[0].tool_results,
            "risk_rejections": rej_conf_res.agent_traces[1].rejection_reasons,
            "events_summary": [e.event for e in rej_conf_res.events],
        },
        "bounded_termination_case": {
            "briefing_status": bounded_res.briefing.status,
            "verified_risk_preserved": bounded_res.briefing.risk.verified_by_agent,
            "abstentions": [a.model_dump() for a in bounded_res.briefing.abstentions],
            "human_review": bounded_res.briefing.human_review.model_dump(),
            "events_summary": [e.event for e in bounded_res.events],
        },
    }


async def capture_live_run() -> dict[str, Any]:
    """Run one complete 4-agent orchestration live against NVIDIA NIM, Pinecone, Snapshot, and Open-Meteo."""
    orch = DisasterMindOrchestrator(settings=get_settings())
    scenario = ScenarioIntakeRequest(
        disaster_type="Storm",
        region="Odisha, India",
        severity="High",
        time_horizon="48 hours",
        notes=(
            "Severe cyclonic storm approaching Odisha coastline. User planning assumption: "
            "exposed coastal population is 120000 people. Local shelter/water inventory has not been provided."
        ),
    )
    res = await orch.run(scenario, run_id="task4-live-odisha-cyclone")
    return res.model_dump()


async def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()

    # 1. Grep audit for json.loads and regex across active backend/
    grep_proc = subprocess.run(
        ["git", "grep", "-n", "-E", r"json\.loads|re\.(search|match|findall|sub|compile)", "--", "backend/"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    (OUT_DIR / "json_loads_regex_audit.txt").write_text(
        f"# git_commit={commit_hash}\n{grep_proc.stdout}",
        encoding="utf-8",
    )

    # 2. Scripted gate captures
    scripted = await capture_scripted_gates()
    (OUT_DIR / "scripted_gates.json").write_text(
        json.dumps({"git_commit": commit_hash, **scripted}, indent=2),
        encoding="utf-8",
    )

    # 3. Live 4-agent run
    live = await capture_live_run()
    (OUT_DIR / "live_briefing.json").write_text(
        json.dumps({"git_commit": commit_hash, "briefing": live["briefing"], "token_usage": live["token_usage"], "total_llm_calls": live["total_llm_calls"], "total_tool_calls": live["total_tool_calls"], "wall_time_ms": live["wall_time_ms"]}, indent=2),
        encoding="utf-8",
    )
    (OUT_DIR / "live_traces.json").write_text(
        json.dumps({"git_commit": commit_hash, "agent_traces": live["agent_traces"]}, indent=2),
        encoding="utf-8",
    )
    (OUT_DIR / "live_events.json").write_text(
        json.dumps({"git_commit": commit_hash, "events": live["events"]}, indent=2),
        encoding="utf-8",
    )

    b = live["briefing"]
    print(
        f"[TASK 4 LIVE DONE] commit={commit_hash[:8]} status={b['status']} "
        f"llm_calls={live['total_llm_calls']} tool_calls={live['total_tool_calls']} "
        f"tokens={live['token_usage']['total_tokens']} wall_ms={live['wall_time_ms']} "
        f"citations={len(b['citations'])} quantities={len(b['quantities'])} "
        f"abstentions={len(b['abstentions'])} conflicts={len(b['conflicts'])}"
    )


if __name__ == "__main__":
    asyncio.run(main())
