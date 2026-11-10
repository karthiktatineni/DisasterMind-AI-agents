"""Offline deterministic Task 4 tests covering 4-agent native tool calling, verifier rejections, preconditions, termination, conflicts, and SSE streaming."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
import httpx
import pytest

from backend.app.config import Settings
from backend.app.main import app
from backend.app.orchestration.models import AGENT_SEQUENCE, Briefing
from backend.app.orchestration.orchestrator import AGENT_SPECS, DisasterMindOrchestrator
from backend.app.schemas import DisasterScenario, RetrievalMatch, RetrievalResult, ScenarioIntakeRequest
from backend.app.services.historical_stats import HistoricalStatsService
from backend.app.services.llm_client import LLMToolResponse, ToolCallRequest


class ScriptedToolLLMClient:
    """Deterministic scripted LLM client implementing NativeToolLLMClient interface for offline tests."""

    def __init__(self, handler) -> None:
        self._handler = handler
        self.calls: list[dict[str, Any]] = []

    async def chat_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] = "required",
        temperature: float = 0.0,
        seed: int = 42,
        max_tokens: int = 1200,
        model: str | None = None,
        **_kwargs: Any,
    ) -> LLMToolResponse:
        assert tools, "Every LLM call must pass non-empty tools JSON schema array."
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "temperature": temperature,
                "seed": seed,
            }
        )
        return self._handler(messages=messages, tools=tools, idx=len(self.calls))

    async def aclose(self) -> None:
        return None


class FakeRetrievalService:
    def __init__(self, *, status: str = "ok") -> None:
        self.status = status

    async def search(self, query_text: str, **_kwargs: Any) -> RetrievalResult:
        if self.status != "ok":
            return RetrievalResult(
                status=self.status,  # type: ignore[arg-type]
                threshold_used=0.3112,
                top_score=0.20,
                namespace="v2-test",
                dataset_version="3d35e5131d79a79d",
                reason="No historical disaster exceeded calibrated similarity threshold.",
            )
        return RetrievalResult(
            status="ok",
            record_ids=["2019-0123-IND", "2013-0401-IND"],
            matches=[
                RetrievalMatch(
                    id="2019-0123-IND",
                    event_name="Cyclone Fani",
                    disaster_type="Storm",
                    country="India",
                    region="Odisha",
                    start_year=2019,
                    total_deaths=89,
                    total_affected=20000000,
                    total_damage=1810000,
                    similarity=0.5812,
                    search_text="CONFIDENTIAL_RAW_SEARCH_TEXT_MUST_NEVER_REACH_MODEL",
                ),
                RetrievalMatch(
                    id="2013-0401-IND",
                    event_name="Cyclone Phailin",
                    disaster_type="Storm",
                    country="India",
                    region="Odisha",
                    start_year=2013,
                    total_deaths=47,
                    total_affected=13230000,
                    total_damage=1500000,
                    similarity=0.5311,
                    search_text="ANOTHER_RAW_SEARCH_TEXT_SNIPPET",
                ),
            ],
            threshold_used=0.3112,
            top_score=0.5812,
            namespace="v2-test",
            dataset_version="3d35e5131d79a79d",
        )


def _make_fake_stats() -> HistoricalStatsService:
    rows = [
        {
            "disaster_id": f"201{i}-0001-IND",
            "disaster_type": "Storm",
            "disaster_subtype": "Tropical cyclone",
            "country": "India",
            "start_year": 2010 + i,
            "total_deaths": 50 + i * 10,
            "total_affected": 1000000 + i * 500000,
            "total_damage_usd_thousands": 500000 + i * 100000,
        }
        for i in range(6)
    ]
    return HistoricalStatsService(records=rows, dataset_version="3d35e5131d79a79d")


def _make_fake_http() -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if "geocoding-api.open-meteo.com" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "name": "Visakhapatnam",
                            "admin1": "Andhra Pradesh",
                            "country": "India",
                            "latitude": 17.6868,
                            "longitude": 83.2185,
                            "population": 1730320,
                        }
                    ]
                },
            )
        if "api.open-meteo.com/v1/forecast" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "current": {
                        "temperature_2m": 31.2,
                        "relative_humidity_2m": 82,
                        "precipitation": 14.5,
                        "wind_speed_10m": 46.0,
                    }
                },
            )
        return httpx.Response(404, json={"error": "unexpected"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _tc(cid: str, name: str, args: dict[str, Any]) -> ToolCallRequest:
    return ToolCallRequest(id=cid, name=name, raw_arguments=json.dumps(args))


def _happy_handler(*, risk_level: str = "high", evac_rec: str = "mandatory_evacuation"):
    def _handle(*, messages: list[dict[str, Any]], tools: list[dict[str, Any]], idx: int) -> LLMToolResponse:
        names = {t["function"]["name"] for t in tools}
        # Route by submission tool in schema allow-list
        if "submit_coordinator_report" in names:
            tool_msgs = [m for m in messages if m.get("role") == "tool"]
            seen = {json.loads(m["content"]).get("agent") for m in tool_msgs if m.get("content", "").startswith("{")}
            if "risk" not in seen:
                tc = _tc("coord_1", "run_risk_agent", {"focus_question": "Assess cyclone risk for Odisha."})
            elif "logistics" not in seen:
                tc = _tc("coord_2", "run_logistics_agent", {"focus_question": "Assess Sphere relief standards."})
            elif "evacuation" not in seen:
                tc = _tc("coord_3", "run_evacuation_agent", {"focus_question": "Recommend evacuation posture."})
            else:
                tc = _tc(
                    "coord_4",
                    "submit_coordinator_report",
                    {
                        "executive_summary": "Verified 4-agent cyclone briefing grounded in Cyclone Fani (2019-0123-IND).",
                        "action_plan": ["Initiate coastal shelter activation", "Confirm commander sign-off"],
                        "cited_record_ids": ["2019-0123-IND"],
                        "data_gaps": ["Shelter capacity not provided by user"],
                    },
                )
            return LLMToolResponse(model="nvidia/nemotron-3-ultra-550b-a55b", request_id=f"req-{idx}", finish_reason="tool_calls", content=None, reasoning_content=None, tool_calls=[tc], prompt_tokens=100, completion_tokens=30, total_tokens=130)

        if "submit_risk_report" in names:
            tool_msgs = [m for m in messages if m.get("role") == "tool"]
            if len(tool_msgs) == 0:
                tcs = [
                    _tc("call_risk_search", "search_precedents", {"query": "Storm Odisha India", "disaster_type": "Storm", "country": "India", "top_k": 5}),
                    _tc("call_risk_stats", "get_historical_stats", {"disaster_type": "Storm", "country": "India"}),
                ]
            else:
                tcs = [
                    _tc(
                        "call_risk_sub",
                        "submit_risk_report",
                        {
                            "risk_level": risk_level,
                            "premise_valid": True,
                            "premise_issue": None,
                            "hazard_summary": "Precedent Cyclone Fani matched with verified historical impact totals.",
                            "cited_record_ids": ["2019-0123-IND"],
                            "quantities": [{"name": "precedent_similarity", "value": 0.5812, "unit": "cosine", "provenance": "call_risk_search", "inputs": []}],
                            "data_gaps": [],
                        },
                    )
                ]
            return LLMToolResponse(model="nvidia/nemotron-3-ultra-550b-a55b", request_id=f"req-{idx}", finish_reason="tool_calls", content=None, reasoning_content=None, tool_calls=tcs, prompt_tokens=90, completion_tokens=25, total_tokens=115)

        if "submit_logistics_report" in names:
            tool_msgs = [m for m in messages if m.get("role") == "tool"]
            if len(tool_msgs) == 0:
                tcs = [
                    _tc("call_log_stats", "get_historical_stats", {"disaster_type": "Storm", "country": "India"}),
                    _tc(
                        "call_log_sphere",
                        "compute_logistics_requirements",
                        {"population": 100000, "population_provenance": "assumption supplied by user"},
                    ),
                ]
            else:
                sphere_payload = json.loads(tool_msgs[-1]["content"])
                tcs = [
                    _tc(
                        "call_log_sub",
                        "submit_logistics_report",
                        {
                            "resource_adequacy": "insufficient_inventory_data",
                            "capacity_comparisons": sphere_payload["capacity_comparisons"],
                            "quantities": sphere_payload["quantities"],
                            "supply_recommendations": ["Verify municipal potable water tanker stocks before deployment."],
                            "cited_record_ids": ["2019-0123-IND"],
                            "data_gaps": ["Shelter and potable water inventory not provided by user"],
                        },
                    )
                ]
            return LLMToolResponse(model="nvidia/nemotron-3-ultra-550b-a55b", request_id=f"req-{idx}", finish_reason="tool_calls", content=None, reasoning_content=None, tool_calls=tcs, prompt_tokens=90, completion_tokens=25, total_tokens=115)

        if "submit_evacuation_report" in names:
            tool_msgs = [m for m in messages if m.get("role") == "tool"]
            if len(tool_msgs) == 0:
                tc = _tc("call_evac_search", "search_precedents", {"query": "Storm evacuation coastal India", "disaster_type": "Storm"})
            else:
                tc = _tc(
                    "call_evac_sub",
                    "submit_evacuation_report",
                    {
                        "recommendation": evac_rec,
                        "target_headcount": 100000,
                        "target_headcount_provenance": "assumption supplied by user",
                        "trigger_criteria": ["High coastal cyclone risk grounded in Cyclone Fani precedent"],
                        "priority_zones": ["Low-lying Odisha coastal belt"],
                        "quantities": [{"name": "target_evacuation_population", "value": 100000, "unit": "people", "provenance": "assumption supplied by user", "inputs": []}],
                        "cited_record_ids": ["2019-0123-IND"],
                        "data_gaps": [],
                    },
                )
            return LLMToolResponse(model="nvidia/nemotron-3-ultra-550b-a55b", request_id=f"req-{idx}", finish_reason="tool_calls", content=None, reasoning_content=None, tool_calls=[tc], prompt_tokens=85, completion_tokens=25, total_tokens=110)

        raise AssertionError(f"Unexpected tool schema set: {names}")

    return _handle


@pytest.mark.asyncio
async def test_task4_full_orchestration_happy_path_and_schema_export():
    settings = Settings(nvidia_api_key="dummy-offline-test", nvidia_model="nvidia/nemotron-3-ultra-550b-a55b")
    llm = ScriptedToolLLMClient(_happy_handler())
    async with _make_fake_http() as http_client:
        orch = DisasterMindOrchestrator(
            settings=settings,
            llm_client=llm,  # type: ignore[arg-type]
            retrieval_service=FakeRetrievalService(),  # type: ignore[arg-type]
            historical_stats_service=_make_fake_stats(),
            http_client=http_client,
        )
        scenario = ScenarioIntakeRequest(disaster_type="Storm", region="Odisha", severity="High", notes="Pop 100k supplied")
        res = await orch.run(scenario, run_id="run-happy-1")

    assert res.briefing.status == "complete"
    assert res.briefing.dataset_version == "3d35e5131d79a79d"
    assert res.briefing.human_review.required is True
    assert res.briefing.conflicts == []
    assert [c.record_id for c in res.briefing.citations] == ["2019-0123-IND"]
    assert res.briefing.logistics_summary is not None
    assert all(c.available is None and c.available_status == "not_provided" for c in res.briefing.logistics_summary.capacity_comparisons)
    assert "CONFIDENTIAL_RAW_SEARCH_TEXT" not in json.dumps(res.model_dump())

    event_types = [e.event for e in res.events]
    for required_ev in ("run_started", "agent_started", "tool_called", "tool_result", "agent_verified", "state_written", "briefing_ready"):
        assert required_ev in event_types

    # Validate exported JSON schema matches Briefing cleanly
    exported = json.loads(Path("docs/briefing_schema.json").read_text(encoding="utf-8"))
    assert exported["title"] == "Briefing"
    assert "human_review" in exported["properties"]
    assert "quantities" in exported["properties"]
    Briefing.model_validate(res.briefing.model_dump())


@pytest.mark.asyncio
async def test_task4_scripted_rejection_cases_and_precondition_and_conflict():
    """Tests malformed args, unknown tool, out-of-allow-list, skipped required tool, precondition, termination, and conflict."""
    settings = Settings(nvidia_api_key="dummy", nvidia_model="nvidia/nemotron-3-ultra-550b-a55b")

    # 1. Precondition + Conflict + All Rejection Modes
    risk_step = {"n": 0}
    coord_step = {"n": 0}

    def rejection_and_conflict_handler(*, messages: list[dict[str, Any]], tools: list[dict[str, Any]], idx: int) -> LLMToolResponse:
        names = {t["function"]["name"] for t in tools}
        if "submit_coordinator_report" in names:
            coord_step["n"] += 1
            step = coord_step["n"]
            if step == 1:
                # Precondition test: attempt Evacuation BEFORE Risk
                tc = _tc("c1", "run_evacuation_agent", {"focus_question": "Evacuate now"})
            elif step == 2:
                # Assert previous tool result reported precondition failure!
                assert "precondition_failed" in messages[-1]["content"]
                tc = _tc("c2", "run_risk_agent", {"focus_question": "Run Risk first"})
            elif step == 3:
                tc = _tc("c3", "run_logistics_agent", {"focus_question": "Run Logistics"})
            elif step == 4:
                tc = _tc("c4", "run_evacuation_agent", {"focus_question": "Now run Evacuation"})
            else:
                tc = _tc(
                    "c5",
                    "submit_coordinator_report",
                    {
                        "executive_summary": "Conflict run completed with Risk vs Evacuation disagreement flagged.",
                        "action_plan": ["Escalate conflict for human review"],
                        "cited_record_ids": ["2019-0123-IND"],
                        "data_gaps": [],
                    },
                )
            return LLMToolResponse(model="test", request_id=f"c-{step}", finish_reason="tool_calls", content=None, reasoning_content=None, tool_calls=[tc])

        if "submit_risk_report" in names:
            risk_step["n"] += 1
            s = risk_step["n"]
            if s == 1:
                # Case A: unknown tool
                tc = _tc("r1", "nonexistent_tool", {"x": 1})
            elif s == 2:
                # Case B: out-of-allow-list tool (compute_logistics_requirements is not allowed for Risk)
                tc = _tc("r2", "compute_logistics_requirements", {"population": 1000, "population_provenance": "user"})
            elif s == 3:
                # Case C: malformed JSON args for search_precedents
                return LLMToolResponse(
                    model="test",
                    request_id="r-3",
                    finish_reason="tool_calls",
                    content=None,
                    reasoning_content=None,
                    tool_calls=[ToolCallRequest(id="r3", name="search_precedents", raw_arguments="{broken json")],
                )
            elif s == 4:
                # Case D: skip required tool search_precedents -> submit directly (must be rejected!)
                tc = _tc(
                    "r4",
                    "submit_risk_report",
                    {"risk_level": "low", "premise_valid": True, "hazard_summary": "Low risk.", "cited_record_ids": [], "quantities": [], "data_gaps": []},
                )
            elif s == 5:
                # Now actually call required tool search_precedents
                tc = _tc("r5", "search_precedents", {"query": "India coastal storm", "disaster_type": "Storm"})
            else:
                # Valid submission reporting "low" risk -> will conflict with "mandatory_evacuation"!
                tc = _tc(
                    "r6",
                    "submit_risk_report",
                    {"risk_level": "low", "premise_valid": True, "hazard_summary": "Observed low risk.", "cited_record_ids": ["2019-0123-IND"], "quantities": [], "data_gaps": []},
                )
            return LLMToolResponse(model="test", request_id=f"r-{s}", finish_reason="tool_calls", content=None, reasoning_content=None, tool_calls=[tc])

        return _happy_handler(risk_level="low", evac_rec="mandatory_evacuation")(messages=messages, tools=tools, idx=idx)

    llm = ScriptedToolLLMClient(rejection_and_conflict_handler)
    async with _make_fake_http() as http_client:
        orch = DisasterMindOrchestrator(
            settings=settings,
            llm_client=llm,  # type: ignore[arg-type]
            retrieval_service=FakeRetrievalService(),  # type: ignore[arg-type]
            historical_stats_service=_make_fake_stats(),
            http_client=http_client,
        )
        res = await orch.run({"disaster_type": "Storm", "region": "Odisha", "severity": "Moderate"}, run_id="run-reject-conflict")

    assert res.briefing.status == "complete"
    # Conflict between low risk and mandatory_evacuation MUST be flagged explicitly!
    assert len(res.briefing.conflicts) == 1
    assert res.briefing.conflicts[0].kind == "risk_evacuation_mismatch"
    assert res.briefing.human_review.required is True
    assert any(e.event == "conflict_flagged" for e in res.events)

    risk_trace = [t for t in res.agent_traces if t.agent == "risk"][0]
    joined_rejects = " | ".join(risk_trace.rejection_reasons)
    assert "Unknown tool 'nonexistent_tool'" in joined_rejects
    assert "outside agent 'risk' allow-list" in joined_rejects
    assert "Malformed arguments for 'search_precedents'" in joined_rejects
    assert "Required tool 'search_precedents' was skipped" in joined_rejects
    assert any(e.event == "agent_rejected" for e in res.events)


@pytest.mark.asyncio
async def test_task4_bounded_termination_returns_partial_briefing():
    """If Coordinator keeps skipping submission past max_steps, return partial Briefing from verified state only."""
    settings = Settings(nvidia_api_key="dummy", nvidia_model="nvidia/nemotron-3-ultra-550b-a55b")
    bounded_specs = {
        k: (
            type(v)(**{**v.__dict__, "max_steps": 2})
            if k == "coordinator"
            else v
        )
        for k, v in AGENT_SPECS.items()
    }

    def loop_forever_coord(*, messages: list[dict[str, Any]], tools: list[dict[str, Any]], idx: int) -> LLMToolResponse:
        names = {t["function"]["name"] for t in tools}
        if "submit_coordinator_report" in names:
            tool_msgs = [m for m in messages if m.get("role") == "tool"]
            if len(tool_msgs) == 0:
                return LLMToolResponse(model="test", request_id="c1", finish_reason="tool_calls", content=None, reasoning_content=None, tool_calls=[_tc("c1", "run_risk_agent", {"focus_question": "Run risk"})])
            # Step 2: Coordinator refuses to submit and sends free text without tool_calls -> hits max_steps=2!
            return LLMToolResponse(model="test", request_id="c2", finish_reason="stop", content="I decline to call a tool", reasoning_content=None, tool_calls=[])
        return _happy_handler()(messages=messages, tools=tools, idx=idx)

    llm = ScriptedToolLLMClient(loop_forever_coord)
    async with _make_fake_http() as http_client:
        orch = DisasterMindOrchestrator(
            settings=settings,
            llm_client=llm,  # type: ignore[arg-type]
            retrieval_service=FakeRetrievalService(),  # type: ignore[arg-type]
            historical_stats_service=_make_fake_stats(),
            http_client=http_client,
            agent_specs=bounded_specs,
        )
        res = await orch.run({"disaster_type": "Storm", "region": "Odisha"}, run_id="run-bounded")

    assert res.briefing.status == "partial"
    assert res.briefing.risk.verified_by_agent == "risk"
    assert res.briefing.evacuation_recommendation.recommendation == "insufficient_data"
    assert any(a.agent == "coordinator" for a in res.briefing.abstentions)
    assert res.briefing.human_review.required is True
    assert any(e.event == "agent_abstained" for e in res.events)


def test_task4_fastapi_sse_endpoint_streams_real_events(monkeypatch):
    """Verify FastAPI /orchestrate/stream emits real SSE transitions without 500 or fallback."""
    from backend.app import main as main_mod

    monkeypatch.setattr(main_mod, "verify_runtime_alignment", lambda *_args, **_kwargs: {"status": "ok"})
    fake_orch = DisasterMindOrchestrator(
        settings=Settings(nvidia_api_key="dummy", nvidia_model="nvidia/nemotron-3-ultra-550b-a55b"),
        llm_client=ScriptedToolLLMClient(_happy_handler()),  # type: ignore[arg-type]
        retrieval_service=FakeRetrievalService(),  # type: ignore[arg-type]
        historical_stats_service=_make_fake_stats(),
        http_client=_make_fake_http(),
    )
    monkeypatch.setattr(main_mod, "_get_orchestrator", lambda: fake_orch)

    with TestClient(app) as client:
        resp = client.post(
            "/orchestrate/stream",
            json={"disaster_type": "Storm", "region": "Odisha", "severity": "High", "notes": "Test stream"},
        )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")
    assert "event: run_started" in resp.text
    assert "event: agent_verified" in resp.text
    assert "event: briefing_ready" in resp.text


@pytest.mark.asyncio
async def test_task4_looping_risk_forced_to_submit_and_deduped():
    """Verify duplicate calls return cached already_provided result and round cap forces named submit_risk_report."""
    settings = Settings(nvidia_api_key="dummy", nvidia_model="nvidia/nemotron-3-ultra-550b-a55b")
    custom_specs = {
        k: (
            type(v)(**{**v.__dict__, "force_submit_on_required_ok": False, "max_tool_rounds": 3})
            if k == "risk"
            else v
        )
        for k, v in AGENT_SPECS.items()
    }
    stats_calls = {"count": 0}
    orig_stats = _make_fake_stats()

    class CountingStats:
        dataset_version = orig_stats.dataset_version

        def query_stats(self, **kwargs: Any) -> dict[str, Any]:
            stats_calls["count"] += 1
            return orig_stats.query_stats(**kwargs)

    risk_turns = {"n": 0}
    seen_choices: list[Any] = []

    class CaptureLLM(ScriptedToolLLMClient):
        async def chat_with_tools(self, **kwargs: Any) -> LLMToolResponse:
            resp = await super().chat_with_tools(**kwargs)
            if any(t["function"]["name"] == "submit_risk_report" for t in kwargs["tools"]):
                seen_choices.append(kwargs["tool_choice"])
            return resp

    def looping_handler(*, messages: list[dict[str, Any]], tools: list[dict[str, Any]], idx: int) -> LLMToolResponse:
        names = {t["function"]["name"] for t in tools}
        if "submit_risk_report" in names:
            risk_turns["n"] += 1
            turn = risk_turns["n"]
            if turn == 1:
                return LLMToolResponse(model="test", request_id="r1", finish_reason="tool_calls", content=None, reasoning_content=None, tool_calls=[_tc("c1", "get_historical_stats", {"disaster_type": "Storm", "country": "India"})])
            if turn == 2:
                # Duplicate identical call -> must dedup with already_provided=true & force submit on next turn!
                return LLMToolResponse(model="test", request_id="r2", finish_reason="tool_calls", content=None, reasoning_content=None, tool_calls=[_tc("c2", "get_historical_stats", {"disaster_type": "Storm", "country": "India"})])
            if turn == 3:
                # Model forced to submit, but hasn't called search_precedents yet; verifier rejects -> resets to let search run or let's call search + submit when forced
                # Actually let's check: when turn 2 deduped, turn 3 received forced tool_choice! Let's submit without search -> rejected -> then search -> submit!
                return LLMToolResponse(model="test", request_id="r3", finish_reason="tool_calls", content=None, reasoning_content=None, tool_calls=[_tc("c3", "submit_risk_report", {"risk_level": "high", "premise_valid": True, "hazard_summary": "Missing search attempt.", "cited_record_ids": [], "quantities": [], "data_gaps": []})])
            if turn == 4:
                return LLMToolResponse(model="test", request_id="r4", finish_reason="tool_calls", content=None, reasoning_content=None, tool_calls=[_tc("c4", "search_precedents", {"query": "Storm Odisha", "disaster_type": "Storm"})])
            return LLMToolResponse(model="test", request_id="r5", finish_reason="tool_calls", content=None, reasoning_content=None, tool_calls=[_tc("c5", "submit_risk_report", {"risk_level": "high", "premise_valid": True, "hazard_summary": "Verified after required search.", "cited_record_ids": ["2019-0123-IND"], "quantities": [], "data_gaps": []})])
        return _happy_handler()(messages=messages, tools=tools, idx=idx)

    llm = CaptureLLM(looping_handler)
    async with _make_fake_http() as http_client:
        orch = DisasterMindOrchestrator(
            settings=settings,
            llm_client=llm,  # type: ignore[arg-type]
            retrieval_service=FakeRetrievalService(),  # type: ignore[arg-type]
            historical_stats_service=CountingStats(),  # type: ignore[arg-type]
            http_client=http_client,
            agent_specs=custom_specs,
        )
        res = await orch.run({"disaster_type": "Storm", "region": "Odisha"}, run_id="run-loop-dedup")

    assert res.briefing.status == "complete"
    # Historical stats executed only once for Risk despite 2 identical calls (+1 once for Logistics = 2 total).
    assert stats_calls["count"] == 2
    assert seen_choices[2] == {"type": "function", "function": {"name": "submit_risk_report"}}
    risk_trace = [t for t in res.agent_traces if t.agent == "risk"][0]
    assert risk_trace.steps[1].tool_results[0].get("deduped") is True
    assert "already_provided" in risk_trace.steps[2].request_messages[-2]["content"]


@pytest.mark.asyncio
async def test_task4_budget_exhaustion_abstained_and_no_redelegation():
    """Exhausted token budget marks agent abstained and Coordinator cannot re-delegate."""
    settings = Settings(nvidia_api_key="dummy", nvidia_model="nvidia/nemotron-3-ultra-550b-a55b")
    tiny_budget_specs = {
        k: (type(v)(**{**v.__dict__, "max_token_budget": 100}) if k == "risk" else v)
        for k, v in AGENT_SPECS.items()
    }
    risk_invocations = {"count": 0}
    coord_step = {"n": 0}

    def budget_handler(*, messages: list[dict[str, Any]], tools: list[dict[str, Any]], idx: int) -> LLMToolResponse:
        names = {t["function"]["name"] for t in tools}
        if "submit_coordinator_report" in names:
            coord_step["n"] += 1
            s = coord_step["n"]
            if s == 1:
                tc = _tc("c1", "run_risk_agent", {"focus_question": "Attempt 1"})
            elif s == 2:
                # Re-delegation after budget exhaustion -> MUST return blocked error!
                tc = _tc("c2", "run_risk_agent", {"focus_question": "Retry after budget exhaustion"})
            elif s == 3:
                assert "Re-delegation blocked" in messages[-1]["content"]
                tc = _tc("c3", "submit_coordinator_report", {"executive_summary": "Partial after Risk budget exhaustion.", "action_plan": [], "cited_record_ids": [], "data_gaps": ["Risk budget exhausted"]})
            else:
                tc = _tc("c4", "submit_coordinator_report", {"executive_summary": "Fallback exit", "action_plan": [], "cited_record_ids": [], "data_gaps": []})
            return LLMToolResponse(model="test", request_id=f"c-{s}", finish_reason="tool_calls", content=None, reasoning_content=None, tool_calls=[tc], total_tokens=50)
        if "submit_risk_report" in names:
            risk_invocations["count"] += 1
            # First step consumes 200 > 100 token budget -> step 2 stops via budget_exhausted!
            return LLMToolResponse(model="test", request_id="r-1", finish_reason="tool_calls", content=None, reasoning_content=None, tool_calls=[_tc("r1", "get_historical_stats", {"disaster_type": "Storm", "country": "India"})], prompt_tokens=150, completion_tokens=50, total_tokens=200)
        return _happy_handler()(messages=messages, tools=tools, idx=idx)

    async with _make_fake_http() as http_client:
        orch = DisasterMindOrchestrator(
            settings=settings,
            llm_client=ScriptedToolLLMClient(budget_handler),  # type: ignore[arg-type]
            retrieval_service=FakeRetrievalService(),  # type: ignore[arg-type]
            historical_stats_service=_make_fake_stats(),
            http_client=http_client,
            agent_specs=tiny_budget_specs,
        )
        res = await orch.run({"disaster_type": "Storm", "region": "Odisha"}, run_id="run-budget-block")

    assert risk_invocations["count"] == 1
    assert any(a.agent == "risk" and "token budget" in a.reason.lower() for a in res.briefing.abstentions)
    assert res.briefing.human_review.required is True


@pytest.mark.asyncio
async def test_task4_geocode_parsing_and_compact_result_cap():
    """Verify geocode handles place+country, single city, unknown place, and compaction bounds tool messages."""
    from backend.app.tools.domain_tools import build_domain_tools
    from backend.app.tools.registry import MAX_TOOL_RESULT_CHARS, ToolRegistry, compact_tool_result_for_model

    def geo_handler(request: httpx.Request) -> httpx.Response:
        name = request.url.params.get("name", "")
        if "," in name:
            return httpx.Response(200, json={})
        if name == "Odisha":
            return httpx.Response(
                200,
                json={"results": [{"name": "Odisha", "country": "India", "country_code": "IN", "latitude": 20.95, "longitude": 85.10, "population": 41974218}]},
            )
        if name == "Chennai":
            return httpx.Response(
                200,
                json={"results": [{"name": "Chennai", "admin1": "Tamil Nadu", "country": "India", "latitude": 13.08, "longitude": 80.27, "population": 4646732}]},
            )
        if name == "Tohoku":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"name": "Tohoku", "country": "United States", "country_code": "US", "latitude": 1.0, "longitude": 1.0},
                        {"name": "Tohoku", "admin1": "Aomori", "country": "Japan", "country_code": "JP", "latitude": 40.82, "longitude": 141.20},
                    ]
                },
            )
        return httpx.Response(200, json={})

    class LargeRetrieval:
        async def search(self, *_a: Any, **_kw: Any) -> RetrievalResult:
            matches = [
                RetrievalMatch(
                    id=f"2020-{i:04d}-IND",
                    event_name=f"Very Long Cyclone Historical Precedent Event Number {i} With Extensive Metadata " * 4,
                    disaster_type="Storm",
                    country="India",
                    region="Odisha Coast",
                    start_year=2020,
                    total_deaths=100 + i,
                    total_affected=500000 + i,
                    total_damage=900000,
                    similarity=0.6 - i * 0.01,
                )
                for i in range(10)
            ]
            return RetrievalResult(status="ok", record_ids=[m.id for m in matches], matches=matches, threshold_used=0.3112, top_score=0.6, namespace="v2")

    async with httpx.AsyncClient(transport=httpx.MockTransport(geo_handler)) as http:
        reg = ToolRegistry(build_domain_tools(retrieval_service=LargeRetrieval(), historical_stats_service=_make_fake_stats(), http_client=http))  # type: ignore[arg-type]
        allow = ("geocode_region", "search_precedents")

        r1 = await reg.execute(call_id="g1", agent="coordinator", tool_name="geocode_region", raw_arguments={"region": "Odisha, India"}, allow_list=allow)
        assert r1.status == "ok" and r1.data["country"] == "India" and abs(r1.data["latitude"] - 20.95) < 1e-6

        r2 = await reg.execute(call_id="g2", agent="coordinator", tool_name="geocode_region", raw_arguments={"region": "Chennai"}, allow_list=allow)
        assert r2.status == "ok" and r2.data["country"] == "India" and abs(r2.data["latitude"] - 13.08) < 1e-6

        r3 = await reg.execute(call_id="g3", agent="coordinator", tool_name="geocode_region", raw_arguments={"region": "Tohoku, Japan"}, allow_list=allow)
        assert r3.status == "ok" and r3.data["country"] == "Japan" and abs(r3.data["latitude"] - 40.82) < 1e-6

        r4 = await reg.execute(call_id="g4", agent="coordinator", tool_name="geocode_region", raw_arguments={"region": "Zyxwvut Nowhereville"}, allow_list=allow)
        assert r4.status == "no_data"

        s_env = await reg.execute(call_id="s1", agent="risk", tool_name="search_precedents", raw_arguments={"query": "cyclone india", "top_k": 10}, allow_list=allow)
        assert s_env.status == "ok" and len(s_env.data["matches"]) == 10
        compact_str = compact_tool_result_for_model(s_env)
        assert len(compact_str) <= MAX_TOOL_RESULT_CHARS
        assert json.loads(compact_str)["status"] == "ok"


