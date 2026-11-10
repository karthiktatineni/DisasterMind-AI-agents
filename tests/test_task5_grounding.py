"""Task 5 tests: fallback removal, strict numeric/prose provenance grounding, citation claim-support, premise checks, human approval override, abstentions, and verifier mutation suite."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from backend.app.config import Settings
from backend.app.orchestration.models import (
    Briefing,
    CitationItem,
    SharedState,
    ToolExecutionEnvelope,
)
from backend.app.orchestration.orchestrator import DisasterMindOrchestrator
from backend.app.orchestration.verifier import (
    InAppVerifier,
    build_allowed_number_set,
    detect_physical_impossibility,
    extract_numbers_from_text,
)
from backend.app.services.llm_client import LLMClientError, LLMToolResponse, ToolCallRequest
from tests.test_task4_orchestration import (
    FakeRetrievalService,
    ScriptedToolLLMClient,
    _happy_handler,
    _make_fake_http,
    _make_fake_stats,
    _tc,
)


def _seeded_state(*, notes: str = "Exposed pop 100000") -> SharedState:
    state = SharedState(
        run_id="run-t5-seed",
        dataset_version="3d35e5131d79a79d",
        scenario_inputs={"disaster_type": "Storm", "region": "Odisha", "country": "India", "notes": notes},
    )
    state.record_tool_execution(
        ToolExecutionEnvelope(
            call_id="call_search_1",
            tool_name="search_precedents",
            agent="risk",
            status="ok",
            arguments={"query": "Storm Odisha India"},
            data={
                "matches": [
                    {
                        "record_id": "2019-0123-IND",
                        "event_name": "Cyclone Fani",
                        "disaster_type": "Storm",
                        "country": "India",
                        "region": "Odisha",
                        "start_year": 2019,
                        "total_deaths": 89,
                        "total_affected": 20000000,
                        "total_damage_usd_thousands": 1810000,
                        "similarity": 0.5812,
                    }
                ],
                "match_count": 1,
                "threshold_used": 0.3112,
            },
        )
    )
    state.record_tool_execution(
        ToolExecutionEnvelope(
            call_id="call_wx_1",
            tool_name="get_weather",
            agent="risk",
            status="ok",
            arguments={"latitude": 20.2, "longitude": 85.8},
            data={"temperature_c": 31.2, "wind_speed_kmh": 46.0, "precipitation_mm": 14.5},
        )
    )
    state.record_tool_execution(
        ToolExecutionEnvelope(
            call_id="call_error_1",
            tool_name="geocode_region",
            agent="coordinator",
            status="error",
            arguments={"region": "bad"},
            data={"fake_error_number": 77777},
            error="Upstream timeout",
        )
    )
    return state


def test_allowed_numbers_and_provenance_source_matching():
    state = _seeded_state()
    allowed = build_allowed_number_set(state)
    assert 100000.0 in allowed
    assert 31.2 in allowed
    assert 0.5812 in allowed
    assert 15.0 in allowed
    assert 77777.0 not in allowed  # error tool results must NEVER enter allowed set

    # Masking prevents verified EM-DAT/call IDs from registering standalone numbers while keeping real dates caught
    assert extract_numbers_from_text(
        "Precedent 2019-0123-IND matched.", masked_tokens={"2019-0123-IND"}
    ) == []
    assert extract_numbers_from_text(
        "Precedent 2019-0123-IND killed 89 people in 1999.", masked_tokens={"2019-0123-IND"}
    ) == [89.0, 1999.0]

    verifier = InAppVerifier()
    # Wrong provenance attribution: citing weather call_id for precedent similarity 0.5812 must fail!
    res = verifier.verify_agent_output(
        agent="risk",
        raw_payload={
            "risk_level": "high",
            "premise_valid": True,
            "hazard_summary": "Storm risk grounded in Cyclone Fani.",
            "cited_record_ids": ["2019-0123-IND"],
            "quantities": [
                {"name": "precedent_similarity", "value": 0.5812, "unit": "cosine", "provenance": "call_wx_1", "inputs": []}
            ],
            "data_gaps": [],
        },
        agent_called_tools=["search_precedents", "get_weather"],
        state=state,
    )
    assert not res.valid
    assert any("not grounded in provenance 'call_wx_1'" in e for e in res.errors)


def test_narrative_prose_number_and_off_by_one_rejection():
    state = _seeded_state()
    verifier = InAppVerifier()

    # Ungrounded integer in narrative text + off-by-one (+1 on 89 -> 90) must be rejected
    for bad_text in (
        "Cyclone Fani caused 90 deaths.",
        "Deploy 42 mobile clinics to Odisha.",
    ):
        res = verifier.verify_agent_output(
            agent="risk",
            raw_payload={
                "risk_level": "high",
                "premise_valid": True,
                "hazard_summary": bad_text,
                "cited_record_ids": ["2019-0123-IND"],
                "quantities": [],
                "data_gaps": [],
            },
            agent_called_tools=["search_precedents"],
            state=state,
        )
        assert not res.valid
        assert any("Ungrounded number" in e for e in res.errors)

    # Exact verbatim numbers in narrative (89, 31.2, 15 L/day) + masked ID succeed
    ok = verifier.verify_agent_output(
        agent="risk",
        raw_payload={
            "risk_level": "high",
            "premise_valid": True,
            "hazard_summary": "Cyclone Fani (2019-0123-IND) recorded 89 deaths; current temp 31.2 C.",
            "cited_record_ids": ["2019-0123-IND"],
            "quantities": [],
            "data_gaps": [],
        },
        agent_called_tools=["search_precedents"],
        state=state,
    )
    assert ok.valid, ok.errors


def test_citation_claim_support_and_premise_guards():
    state = _seeded_state()
    # Add an unrelated retrieved record (Earthquake in Chile, low similarity & wrong hazard+country)
    state.retrieved_records["1995-9999-CHL"] = CitationItem(
        record_id="1995-9999-CHL",
        event_name="Antofagasta Earthquake",
        disaster_type="Earthquake",
        country="Chile",
        region="Antofagasta",
        start_year=1995,
        total_deaths=3,
        total_affected=2000,
        similarity=0.45,
    )
    state.retrieved_records["2020-0001-IND"] = CitationItem(
        record_id="2020-0001-IND",
        event_name="Weak Match",
        disaster_type="Storm",
        country="India",
        start_year=2020,
        similarity=0.25,
    )
    verifier = InAppVerifier()

    # Cross-hazard + cross-country irrelevant record must fail claim support
    bad_support = verifier.verify_agent_output(
        agent="risk",
        raw_payload={
            "risk_level": "moderate",
            "premise_valid": True,
            "hazard_summary": "Unrelated citation check.",
            "cited_record_ids": ["1995-9999-CHL"],
            "quantities": [],
            "data_gaps": [],
        },
        agent_called_tools=["search_precedents"],
        state=state,
    )
    assert not bad_support.valid
    assert any("does not support scenario hazard/region" in e for e in bad_support.errors)

    # Below-threshold citation must fail
    low_sim = verifier.verify_agent_output(
        agent="risk",
        raw_payload={
            "risk_level": "moderate",
            "premise_valid": True,
            "hazard_summary": "Below threshold check.",
            "cited_record_ids": ["2020-0001-IND"],
            "quantities": [],
            "data_gaps": [],
        },
        agent_called_tools=["search_precedents"],
        state=state,
    )
    assert not low_sim.valid
    assert any("below calibrated threshold" in e for e in low_sim.errors)

    # Physically impossible scenario cannot be submitted with premise_valid=True
    imp_state = SharedState(
        run_id="run-imp",
        dataset_version="3d35e5131d79a79d",
        scenario_inputs={"disaster_type": "Category 6 Cyclone", "region": "Odisha", "notes": "Mw 11.2 earthquake"},
    )
    assert detect_physical_impossibility(imp_state.scenario_inputs) is not None
    imp_bad = verifier.verify_agent_output(
        agent="risk",
        raw_payload={
            "risk_level": "extreme",
            "premise_valid": True,
            "hazard_summary": "Normal storm.",
            "cited_record_ids": [],
            "quantities": [],
            "data_gaps": [],
        },
        agent_called_tools=["search_precedents"],
        state=imp_state,
    )
    assert not imp_bad.valid
    assert any("Physical impossibility detected" in e for e in imp_bad.errors)

    # Proper premise_valid=False submission passes cleanly with insufficient_data and [] citations
    imp_ok = verifier.verify_agent_output(
        agent="risk",
        raw_payload={
            "risk_level": "insufficient_data",
            "premise_valid": False,
            "premise_issue": "Category 6 cyclone and Mw 11.2 exceed physical bounds.",
            "hazard_summary": "Physically impossible scenario parameters rejected.",
            "cited_record_ids": [],
            "quantities": [],
            "data_gaps": ["No precedents validate impossible physical parameters."],
        },
        agent_called_tools=["search_precedents"],
        state=imp_state,
    )
    assert imp_ok.valid, imp_ok.errors


@pytest.mark.asyncio
@pytest.mark.parametrize("evac_rec", ["mandatory_evacuation", "no_evacuation"])
async def test_requires_human_approval_always_overridden_true_by_code(evac_rec: str):
    """Even when Evacuation model explicitly emits requires_human_approval=false, code overrides it to True."""
    settings = Settings(nvidia_api_key="dummy", nvidia_model="nvidia/nemotron-3-ultra-550b-a55b")
    base_handler = _happy_handler(risk_level="low" if evac_rec == "no_evacuation" else "high", evac_rec=evac_rec)

    def false_approval_handler(*, messages: list[dict[str, Any]], tools: list[dict[str, Any]], idx: int) -> LLMToolResponse:
        resp = base_handler(messages=messages, tools=tools, idx=idx)
        new_calls: list[ToolCallRequest] = []
        for tc in resp.tool_calls:
            if tc.name == "submit_evacuation_report":
                payload = json.loads(tc.raw_arguments)
                payload["requires_human_approval"] = False
                new_calls.append(ToolCallRequest(id=tc.id, name=tc.name, raw_arguments=json.dumps(payload)))
            else:
                new_calls.append(tc)
        return LLMToolResponse(
            model=resp.model,
            request_id=resp.request_id,
            finish_reason=resp.finish_reason,
            content=resp.content,
            reasoning_content=resp.reasoning_content,
            tool_calls=new_calls,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            reasoning_tokens=resp.reasoning_tokens,
            total_tokens=resp.total_tokens,
            latency_ms=resp.latency_ms,
        )

    async with _make_fake_http() as http_client:
        orch = DisasterMindOrchestrator(
            settings=settings,
            llm_client=ScriptedToolLLMClient(false_approval_handler),  # type: ignore[arg-type]
            retrieval_service=FakeRetrievalService(),  # type: ignore[arg-type]
            historical_stats_service=_make_fake_stats(),
            http_client=http_client,
        )
        res = await orch.run(
            {"disaster_type": "Storm", "region": "Odisha", "notes": "Pop 100k supplied"},
            run_id=f"run-hr-{evac_rec}",
            mode_label="historical_precedent" if evac_rec == "no_evacuation" else "hypothetical_scenario",
        )

    assert res.briefing.evacuation_recommendation.requires_human_approval is True
    assert res.briefing.human_review.required is True
    assert any("overridden to true by code" in r.lower() for r in res.briefing.human_review.reasons)
    assert len(res.briefing.disclaimers) >= 1
    assert any("scenario planning" in d.lower() for d in res.briefing.disclaimers)
    expected_label = "historical_precedent" if evac_rec == "no_evacuation" else "hypothetical_scenario"
    assert res.briefing.label == expected_label
    Briefing.model_validate(res.briefing.model_dump())


@pytest.mark.asyncio
async def test_abstention_runs_on_llm_unavailable_and_tool_error():
    """LLM outage, missing key, or failing retrieval+weather must abstain cleanly with zero invented quantities."""
    settings_missing = Settings(nvidia_api_key="", nvidia_model="nvidia/nemotron-3-ultra-550b-a55b")
    async with _make_fake_http() as http_client:
        # 1. Missing NVIDIA key at startup
        orch_no_key = DisasterMindOrchestrator(
            settings=settings_missing,
            retrieval_service=FakeRetrievalService(status="no_data"),  # type: ignore[arg-type]
            historical_stats_service=_make_fake_stats(),
            http_client=http_client,
        )
        res_missing = await orch_no_key.run({"disaster_type": "Flood", "region": "Assam"}, run_id="run-nokey")
        assert res_missing.briefing.status == "insufficient_data"
        assert res_missing.briefing.quantities == []
        assert len(res_missing.briefing.abstentions) >= 1

        # 2. LLM error + tool outages mid-run
        class FailingLLM:
            async def chat_with_tools(self, **_kw: Any) -> LLMToolResponse:
                raise LLMClientError("Simulated NVIDIA 503 outage")

            async def aclose(self) -> None:
                return None

        def failing_http_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, json={"error": "upstream unavailable"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(failing_http_handler)) as err_http:
            orch_err = DisasterMindOrchestrator(
                settings=Settings(nvidia_api_key="dummy"),
                llm_client=FailingLLM(),  # type: ignore[arg-type]
                retrieval_service=FakeRetrievalService(status="error"),  # type: ignore[arg-type]
                historical_stats_service=None,
                http_client=err_http,
            )
            res_err = await orch_err.run({"disaster_type": "Flood", "region": "Assam"}, run_id="run-llm-err")
        assert res_err.briefing.status == "insufficient_data"
        assert res_err.briefing.quantities == []
        assert res_err.briefing.citations == []
        assert any(a.agent == "coordinator" for a in res_err.briefing.abstentions)


def test_task5_verifier_mutations_must_all_be_caught(monkeypatch):
    """Mutation test: deliberately weakening any single verifier guard allows invalid data that must be rejected."""
    from backend.app.orchestration import verifier as vmod

    state = _seeded_state()
    state.retrieved_records["1995-9999-CHL"] = CitationItem(
        record_id="1995-9999-CHL",
        event_name="Chile Quake",
        disaster_type="Earthquake",
        country="Chile",
        start_year=1995,
        similarity=0.50,
    )
    real_verifier = InAppVerifier()

    # Mutant 1: skips narrative prose number extraction
    prose_attack = {
        "risk_level": "high",
        "premise_valid": True,
        "hazard_summary": "Estimated 987654 casualties expected tomorrow.",
        "cited_record_ids": ["2019-0123-IND"],
        "quantities": [],
        "data_gaps": [],
    }
    assert not real_verifier.verify_agent_output(agent="risk", raw_payload=prose_attack, agent_called_tools=["search_precedents"], state=state).valid
    with monkeypatch.context() as mp:
        mp.setattr(vmod, "_extract_narrative_strings", lambda _v: [])
        assert InAppVerifier().verify_agent_output(agent="risk", raw_payload=prose_attack, agent_called_tools=["search_precedents"], state=state).valid

    # Mutant 2: trusts provenance label without checking numeric value attribution
    qty_attack = {
        "risk_level": "high",
        "premise_valid": True,
        "hazard_summary": "Verified storm risk.",
        "cited_record_ids": ["2019-0123-IND"],
        "quantities": [{"name": "invented_deaths", "value": 424242, "unit": "people", "provenance": "call_search_1", "inputs": []}],
        "data_gaps": [],
    }
    assert not real_verifier.verify_agent_output(agent="risk", raw_payload=qty_attack, agent_called_tools=["search_precedents"], state=state).valid
    with monkeypatch.context() as mp:
        mp.setattr(vmod, "_numbers_authorized_by_provenance", lambda *_a, **_kw: {424242.0})
        mp.setattr(vmod, "build_allowed_number_set", lambda *_a, **_kw: {424242.0})
        assert InAppVerifier().verify_agent_output(agent="risk", raw_payload=qty_attack, agent_called_tools=["search_precedents"], state=state).valid

    # Mutant 3: widens numeric tolerance so +1 integer mutation (89 -> 90) slips past
    off_by_one_attack = {
        "risk_level": "high",
        "premise_valid": True,
        "hazard_summary": "Cyclone Fani recorded 90 deaths.",
        "cited_record_ids": ["2019-0123-IND"],
        "quantities": [],
        "data_gaps": [],
    }
    assert not real_verifier.verify_agent_output(agent="risk", raw_payload=off_by_one_attack, agent_called_tools=["search_precedents"], state=state).valid
    with monkeypatch.context() as mp:
        mp.setattr(vmod, "number_in_allowed", lambda val, allowed: any(abs(float(val) - a) <= 2.0 for a in allowed))
        assert InAppVerifier().verify_agent_output(agent="risk", raw_payload=off_by_one_attack, agent_called_tools=["search_precedents"], state=state).valid

    # Mutant 4: skips citation claim-support check (only checks ID existence)
    class MutantNoClaimSupport(InAppVerifier):
        def _verify_citations_support_claim(self, *, agent, validated, state, errors):
            for rid in getattr(validated, "cited_record_ids", []):
                if rid not in state.retrieved_records:
                    errors.append(f"Missing {rid}")

    bad_cite_attack = {
        "risk_level": "moderate",
        "premise_valid": True,
        "hazard_summary": "Citing Chile quake for India storm.",
        "cited_record_ids": ["1995-9999-CHL"],
        "quantities": [],
        "data_gaps": [],
    }
    assert not real_verifier.verify_agent_output(agent="risk", raw_payload=bad_cite_attack, agent_called_tools=["search_precedents"], state=state).valid
    assert MutantNoClaimSupport().verify_agent_output(agent="risk", raw_payload=bad_cite_attack, agent_called_tools=["search_precedents"], state=state).valid

    # Mutant 5: skips physical impossibility check
    imp_state = _seeded_state(notes="")
    imp_state.scenario_inputs = {"disaster_type": "Category 6 Cyclone", "region": "Odisha", "country": "India"}
    imp_attack = {
        "risk_level": "high",
        "premise_valid": True,
        "hazard_summary": "Accepted impossible cyclone.",
        "cited_record_ids": [],
        "quantities": [],
        "data_gaps": [],
    }
    assert not real_verifier.verify_agent_output(agent="risk", raw_payload=imp_attack, agent_called_tools=["search_precedents"], state=imp_state).valid
    with monkeypatch.context() as mp:
        mp.setattr(vmod, "detect_physical_impossibility", lambda _inputs: None)
        assert InAppVerifier().verify_agent_output(agent="risk", raw_payload=imp_attack, agent_called_tools=["search_precedents"], state=imp_state).valid
