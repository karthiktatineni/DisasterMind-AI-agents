"""Task 4 live & scripted gate verifier: runs 3x strong-precedent flood + storm w/ coords + no-precedent + impossible scenarios."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import get_settings
from backend.app.orchestration.models import OrchestrationRunResult, SSEEventRecord
from backend.app.orchestration.orchestrator import AGENT_SPECS, DisasterMindOrchestrator
from backend.app.schemas import DisasterScenario, ScenarioIntakeRequest
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
RAW_DIR = OUT_DIR / "raw_traces"


def _redacted_ordered_trace(res: OrchestrationRunResult) -> list[dict[str, Any]]:
    """Produce compact redacted ordered trace showing request tools, tool_calls, results, verification per agent."""
    ordered: list[dict[str, Any]] = []
    for trace in res.agent_traces:
        steps_view = []
        for s in trace.steps:
            steps_view.append(
                {
                    "step": s.step,
                    "tools_sent": s.tools_sent,
                    "request_roles": [m.get("role") for m in s.request_messages],
                    "tool_calls_received": [
                        {"id": tc["id"], "name": tc["name"], "arguments_len": len(str(tc.get("arguments", "")))}
                        for tc in s.tool_calls_received
                    ],
                    "tool_results": [
                        {
                            "call_id": r.get("call_id"),
                            "tool_name": r.get("tool_name"),
                            "status": r.get("status"),
                            "deduped": bool(r.get("deduped")),
                            "latency_ms": r.get("latency_ms"),
                            "error": r.get("error"),
                        }
                        for r in s.tool_results
                    ],
                    "verification_passed": s.verification_passed,
                    "verification_errors": s.verification_errors,
                    "latency_ms": s.latency_ms,
                    "tokens": s.usage.total_tokens,
                }
            )
        ordered.append(
            {
                "agent": trace.agent,
                "status": trace.status,
                "called_tools": trace.called_tools,
                "rejection_reasons": trace.rejection_reasons,
                "total_latency_ms": trace.total_latency_ms,
                "usage": trace.usage.model_dump(),
                "steps": steps_view,
            }
        )
    return ordered


async def capture_scripted_gates() -> dict[str, Any]:
    """Capture explicit scripted gate traces: rejections, precondition, bounded termination, conflict."""
    settings = get_settings()
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


SCENARIOS: list[tuple[str, DisasterScenario | ScenarioIntakeRequest]] = [
    (
        "flood_strong_precedent_run1",
        ScenarioIntakeRequest(
            disaster_type="Flood",
            region="Bangladesh",
            severity="High",
            time_horizon="48 hours",
            notes="Severe monsoon riverine flood along Brahmaputra basin. User assumption: 150000 exposed people. Shelter and water inventory not provided.",
        ),
    ),
    (
        "flood_strong_precedent_run2",
        ScenarioIntakeRequest(
            disaster_type="Flood",
            region="Bangladesh",
            severity="High",
            time_horizon="48 hours",
            notes="Severe monsoon riverine flood along Brahmaputra basin. User assumption: 150000 exposed people. Shelter and water inventory not provided.",
        ),
    ),
    (
        "flood_strong_precedent_run3",
        ScenarioIntakeRequest(
            disaster_type="Flood",
            region="Bangladesh",
            severity="High",
            time_horizon="48 hours",
            notes="Severe monsoon riverine flood along Brahmaputra basin. User assumption: 150000 exposed people. Shelter and water inventory not provided.",
        ),
    ),
    (
        "storm_with_coordinates",
        DisasterScenario(
            disaster_type="Storm",
            region="Visakhapatnam, India",
            country="India",
            latitude=17.6868,
            longitude=83.2185,
            severity="High",
            time_horizon="48 hours",
            population=120000,
            shelter_capacity=30000,
            notes="Severe cyclonic storm approaching Visakhapatnam coast. User-supplied population 120000 and shelter capacity 30000 people; potable water inventory not provided.",
        ),
    ),
    (
        "no_precedent_scenario",
        ScenarioIntakeRequest(
            disaster_type="Meteorite Impact",
            region="Antarctica",
            severity="Critical",
            time_horizon="12 hours",
            notes="Hypothetical bolide impact across interior East Antarctic plateau. No historical disaster analog supplied.",
        ),
    ),
    (
        "impossible_scenario",
        DisasterScenario(
            disaster_type="Earthquake",
            region="Tokyo, Japan",
            country="Japan",
            latitude=35.6762,
            longitude=139.6503,
            severity="Critical",
            time_horizon="24 hours",
            notes="Physically impossible scenario test: simultaneous Category 6 cyclone (420 km/h) and Moment Magnitude 11.5 earthquake.",
        ),
    ),
]


async def run_live_suite(commit_hash: str, *, only_tag: str | None = None, per_scenario_timeout_s: float = 240.0) -> dict[str, Any]:
    """Run Task 4 live scenarios cleanly with timestamped real-time event logging."""
    orch = DisasterMindOrchestrator(settings=get_settings())
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    selected = [(t, s) for (t, s) in SCENARIOS if (only_tag is None or t == only_tag)]
    if only_tag is not None and not selected:
        raise ValueError(f"Unknown scenario tag '{only_tag}'.")

    for tag, scen in selected:
        t0 = time.monotonic()
        print(f"[LIVE RUN START] {tag} (timeout={per_scenario_timeout_s}s)...", flush=True)

        def _on_ev(ev: SSEEventRecord, _tag: str = tag, _t0: float = t0) -> None:
            dt = round(time.monotonic() - _t0, 1)
            print(f"  [+{dt:5.1f}s] [{_tag}] {ev.event} agent={ev.agent} data={json.dumps(ev.data)[:160]}", flush=True)

        res = await asyncio.wait_for(orch.run(scen, run_id=f"task4-{tag}", on_event=_on_ev), timeout=per_scenario_timeout_s)
        RAW_DIR.joinpath(f"{tag}.json").write_text(json.dumps(res.model_dump(), indent=2), encoding="utf-8")
        b = res.briefing
        geocode_used = any(
            r.get("tool_name") == "geocode_region"
            for tr in res.agent_traces
            for st in tr.steps
            for r in st.tool_results
        )
        summary_entry = {
            "git_commit": commit_hash,
            "scenario_tag": tag,
            "run_id": b.run_id,
            "status": b.status,
            "llm_calls": res.total_llm_calls,
            "tool_calls": res.total_tool_calls,
            "tokens": res.token_usage.model_dump(),
            "wall_ms": res.wall_time_ms,
            "geocode_used": geocode_used,
            "ordered_trace": _redacted_ordered_trace(res),
            "briefing": b.model_dump(),
        }
        RAW_DIR.joinpath(f"{tag}_summary.json").write_text(json.dumps(summary_entry, indent=2), encoding="utf-8")
        print(
            f"[LIVE RUN DONE] {tag}: status={b.status} risk={b.risk.level} premise_valid={b.risk.premise_valid} "
            f"evac={b.evacuation_recommendation.recommendation} llm={res.total_llm_calls} "
            f"tools={res.total_tool_calls} toks={res.token_usage.total_tokens} ms={res.wall_time_ms} "
            f"cits={len(b.citations)} qty={len(b.quantities)} abst={len(b.abstentions)} conf={len(b.conflicts)}",
            flush=True,
        )

    return assemble_live_report(commit_hash)


def assemble_live_report(commit_hash: str) -> dict[str, Any]:
    runs_summary: list[dict[str, Any]] = []
    for tag, _ in SCENARIOS:
        p = RAW_DIR / f"{tag}_summary.json"
        if p.exists():
            runs_summary.append(json.loads(p.read_text(encoding="utf-8")))

    flood_runs = [r for r in runs_summary if r["scenario_tag"].startswith("flood_strong_precedent_")]
    flood_agreement = {
        "count": len(flood_runs),
        "statuses": [r["briefing"]["status"] for r in flood_runs],
        "risk_levels": [r["briefing"]["risk"]["level"] for r in flood_runs],
        "evacuation_recommendations": [r["briefing"]["evacuation_recommendation"]["recommendation"] for r in flood_runs],
        "citation_id_sets": [[c["record_id"] for c in r["briefing"]["citations"]] for r in flood_runs],
        "exact_risk_agreement": len({r["briefing"]["risk"]["level"] for r in flood_runs}) == 1 if flood_runs else False,
        "exact_evac_agreement": len({r["briefing"]["evacuation_recommendation"]["recommendation"] for r in flood_runs}) == 1 if flood_runs else False,
    }
    report = {
        "git_commit": commit_hash,
        "completed_runs": len(runs_summary),
        "total_live_llm_calls": sum(r["llm_calls"] for r in runs_summary),
        "flood_repeat_agreement": flood_agreement,
        "runs": runs_summary,
    }
    (OUT_DIR / "live_proof_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None, help="Run a single scenario tag from SCENARIOS.")
    parser.add_argument("--timeout", type=float, default=240.0, help="Per-scenario wall-clock timeout seconds.")
    parser.add_argument("--skip-scripted", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()

    grep_proc = subprocess.run(
        ["git", "grep", "-n", "-E", r"json\.loads|re\.(search|match|findall|sub|compile)", "--", "backend/"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    (OUT_DIR / "json_loads_regex_audit.txt").write_text(f"# git_commit={commit_hash}\n{grep_proc.stdout}", encoding="utf-8")

    div_grep = subprocess.run(
        ["git", "grep", "-n", "-E", r"/ 5000|\* 3|\* 2|186000|4200|132000|_fallback_population", "--", "backend/", "config/"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    (OUT_DIR / "unsourced_divisors_audit.txt").write_text(f"# git_commit={commit_hash}\n{div_grep.stdout}", encoding="utf-8")

    if not args.skip_scripted:
        scripted = await capture_scripted_gates()
        (OUT_DIR / "scripted_gates.json").write_text(
            json.dumps({"git_commit": commit_hash, **scripted}, indent=2),
            encoding="utf-8",
        )

    await run_live_suite(commit_hash, only_tag=args.only, per_scenario_timeout_s=args.timeout)


if __name__ == "__main__":
    asyncio.run(main())
