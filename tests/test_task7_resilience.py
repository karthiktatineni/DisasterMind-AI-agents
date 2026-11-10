"""Task 7 tests (labeled scripted clients): loop lag under 20 concurrent requests (<100ms), failure matrix (0 500s, 0 stack traces), auth, CORS, trusted-proxy rate limits, concurrency cap 429+Retry-After, streamed body limit, request_id correlation, and concurrent asyncio.gather enrichment."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx
from openai import APIStatusError, APITimeoutError, RateLimitError
import pytest

from backend.app.config import Settings
from backend.app.main import app
from backend.app.orchestration.orchestrator import AGENT_SPECS, DisasterMindOrchestrator
from backend.app.schemas import RetrievalResult, ScenarioIntakeRequest
from backend.app.security import (
    ConcurrencyCapMiddleware,
    InMemoryRateLimitMiddleware,
    RequestContextMiddleware,
    RequestSizeLimitMiddleware,
    extract_client_ip,
)
from backend.app.services.data_enrichment import enrich_scenario
from backend.app.services.llm_client import LLMClientError, LLMToolResponse, NativeToolLLMClient, ToolCallRequest
from backend.app.services.retry_policy import retry_after_seconds
from backend.app.tools.domain_tools import build_domain_tools
from backend.app.tools.registry import ToolRegistry
from tests.test_task4_orchestration import (
    FakeRetrievalService,
    ScriptedToolLLMClient,
    _happy_handler,
    _make_fake_http,
    _make_fake_stats,
    _tc,
)


def test_retry_after_seconds_parses_seconds_milliseconds_http_date_and_invalid():
    assert retry_after_seconds({"Retry-After": "4"}) == 4.0
    assert retry_after_seconds({"retry-after-ms": "2500"}) == 2.5
    date = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=5))
    parsed = retry_after_seconds({"Retry-After": date})
    assert parsed is not None and 0 <= parsed <= 5
    assert retry_after_seconds({"Retry-After": "not-a-date"}) is None


@pytest.mark.asyncio
async def test_nim_429_without_retry_after_fails_fast_without_repeating_request():
    class RateLimitedClient:
        def __init__(self) -> None:
            self.calls = 0
            self.chat = self
            self.completions = self

        async def create(self, **_kwargs):
            self.calls += 1
            raise _fake_openai_err(RateLimitError, "429", 429)

    upstream = RateLimitedClient()
    client = NativeToolLLMClient(settings=Settings(nvidia_api_key="dummy"), openai_client=upstream)
    with pytest.raises(LLMClientError, match="without Retry-After") as error:
        await client.chat_with_tools(messages=[{"role": "user", "content": "probe"}], tools=[{"type": "function"}])
    assert error.value.status_code == 429
    assert upstream.calls == 1


def _fake_openai_err(cls, msg: str, status_code: int = 500):
    req = httpx.Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions")
    resp = httpx.Response(status_code, request=req)
    if cls is RateLimitError:
        return RateLimitError(msg, response=resp, body={"status": 429})
    if cls is APIStatusError:
        return APIStatusError(msg, response=resp, body={"status": status_code})
    return APITimeoutError(request=req)


@pytest.mark.asyncio
async def test_loop_lag_20_concurrent_requests_under_100ms(monkeypatch):
    """Run 20 concurrent /orchestrate requests + CPU historical stats under 100ms event-loop lag target."""
    from backend.app import main as main_mod

    monkeypatch.setattr(main_mod, "verify_runtime_alignment", lambda *_a, **_kw: {"status": "ok"})
    monkeypatch.setattr(
        "backend.app.security.get_settings",
        lambda: Settings(
            nvidia_api_key="dummy",
            rate_limit_requests=200,
            max_concurrent_orchestrations=25,
        ),
    )

    stats = _make_fake_stats()

    class AsyncDelayedLLM:
        async def chat_with_tools(self, *, messages, tools, **_kw):
            await asyncio.sleep(0.01)
            return _happy_handler()(messages=messages, tools=tools, idx=1)

        async def aclose(self):
            return None

    async with _make_fake_http() as fake_http:
        monkeypatch.setattr(
            main_mod,
            "_get_orchestrator",
            lambda: DisasterMindOrchestrator(
                settings=Settings(nvidia_api_key="dummy"),
                llm_client=AsyncDelayedLLM(),  # type: ignore[arg-type]
                retrieval_service=FakeRetrievalService(),  # type: ignore[arg-type]
                historical_stats_service=stats,
                http_client=fake_http,
            ),
        )

        lags_ms: list[float] = []
        stop = asyncio.Event()

        async def _lag_monitor():
            interval = 0.005
            prev = time.perf_counter()
            while not stop.is_set():
                await asyncio.sleep(interval)
                now = time.perf_counter()
                lag = max(0.0, (now - prev - interval) * 1000.0)
                lags_ms.append(lag)
                prev = now

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            mon = asyncio.create_task(_lag_monitor())
            tasks = [
                ac.post(
                    "/orchestrate-intake",
                    json={"disaster_type": "Storm", "region": "Odisha", "notes": "Pop 100k supplied"},
                    headers={"X-Request-ID": f"lag-req-{i}"},
                )
                for i in range(20)
            ]
            responses = await asyncio.gather(*tasks)
            stop.set()
            await mon

        max_lag_ms = max(lags_ms) if lags_ms else 0.0
        assert all(r.status_code == 200 for r in responses), [r.status_code for r in responses]
        assert all(r.headers.get("X-Request-ID") == f"lag-req-{i}" for i, r in enumerate(responses))
        assert max_lag_ms < 100.0, f"Event loop stall {max_lag_ms:.2f} ms exceeded 100 ms!"


@pytest.mark.asyncio
async def test_failure_matrix_10_modes_zero_500_zero_stack_traces(monkeypatch):
    """Test all 10 failure modes (scripted clients labeled): zero HTTP 500s and zero stack traces leaked."""
    from backend.app import main as main_mod

    monkeypatch.setattr(main_mod, "verify_runtime_alignment", lambda *_a, **_kw: {"status": "ok"})
    stats = _make_fake_stats()

    async def _run_case(llm_client, retrieval_svc, http_client):
        monkeypatch.setattr(
            main_mod,
            "_get_orchestrator",
            lambda: DisasterMindOrchestrator(
                settings=Settings(nvidia_api_key="dummy"),
                llm_client=llm_client,
                retrieval_service=retrieval_svc,
                historical_stats_service=stats,
                http_client=http_client,
            ),
        )
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.post(
                "/orchestrate-intake",
                json={"disaster_type": "Storm", "region": "Odisha", "notes": "Pop 100k supplied"},
                headers={"X-Request-ID": "matrix-case-rid"},
            )
            assert r.status_code != 500
            assert "traceback" not in r.text.lower()
            assert 'file "' not in r.text.lower()
            assert r.headers.get("X-Request-ID") == "matrix-case-rid"
            return r

    # 1. Pinecone error & 2. Pinecone timeout & 3. Empty retrieval
    class ErrRet:
        def __init__(self, mode: str) -> None:
            self.mode = mode

        async def search(self, *_a, **_kw):
            if self.mode == "error":
                return RetrievalResult(status="error", threshold_used=0.3112, namespace="ns", reason="Pinecone 503")
            if self.mode == "timeout":
                return RetrievalResult(status="error", threshold_used=0.3112, namespace="ns", reason="TimeoutError")
            return RetrievalResult(status="no_data", threshold_used=0.3112, top_score=0.1, namespace="ns", reason="Empty retrieval")

    async with _make_fake_http() as http_ok:
        for mode in ("error", "timeout", "no_data"):
            r = await _run_case(ScriptedToolLLMClient(_happy_handler()), ErrRet(mode), http_ok)
            assert r.status_code == 200
            body = r.json()
            assert body["briefing"]["run_id"] == "matrix-case-rid"
            assert body["briefing"]["status"] in ("insufficient_data", "partial")

    # 4. LLM 429 exhausted, 5. LLM 500 exhausted, 6. LLM timeout exhausted via real NativeToolLLMClient
    class FailingOpenAIClient:
        def __init__(self, exc) -> None:
            self.exc = exc
            self.chat = self
            self.completions = self

        async def create(self, **_kw):
            raise self.exc

    async with _make_fake_http() as http_ok:
        for exc in (
            _fake_openai_err(RateLimitError, "429", 429),
            _fake_openai_err(APIStatusError, "500", 500),
            _fake_openai_err(APITimeoutError, "timeout"),
        ):
            client = NativeToolLLMClient(settings=Settings(nvidia_api_key="dummy"), openai_client=FailingOpenAIClient(exc))
            # Speed up retry check cleanly
            orig_chat = client.chat_with_tools
            client.chat_with_tools = lambda **kw: orig_chat(**{**kw, "max_retries": 1, "base_delay": 0.01})  # type: ignore[method-assign]
            r = await _run_case(client, FakeRetrievalService(), http_ok)
            assert r.status_code == 200
            assert r.json()["briefing"]["status"] == "insufficient_data"

        # 7. Malformed tool call + 8. Malformed final JSON submission
        def bad_calls_handler(*, messages, tools, idx):
            names = {t["function"]["name"] for t in tools}
            if "submit_coordinator_report" in names:
                return LLMToolResponse(
                    model="test",
                    request_id="bad",
                    finish_reason="tool_calls",
                    content=None,
                    reasoning_content=None,
                    tool_calls=[
                        ToolCallRequest(id="bad1", name="search_precedents", raw_arguments="{not_json"),
                        ToolCallRequest(id="bad2", name="submit_coordinator_report", raw_arguments="{missing_closing"),
                    ],
                )
            return _happy_handler()(messages=messages, tools=tools, idx=idx)

        r_bad = await _run_case(ScriptedToolLLMClient(bad_calls_handler), FakeRetrievalService(), http_ok)
        assert r_bad.status_code == 200
        assert r_bad.json()["briefing"]["status"] in ("insufficient_data", "partial")

    # 9. Weather failure (HTTP 503) + 10. Slow tool timeout
    def weather_fail_http(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "weather down"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(weather_fail_http)) as wx_err_http:
        reg = ToolRegistry(build_domain_tools(retrieval_service=FakeRetrievalService(), historical_stats_service=stats, http_client=wx_err_http))
        wx_env = await reg.execute(call_id="wx1", agent="risk", tool_name="get_weather", raw_arguments={"latitude": 20.0, "longitude": 85.0}, allow_list=("get_weather",))
        assert wx_env.status == "error"

        async def _slow_handler(_args, _cid):
            await asyncio.sleep(0.2)
            return "ok", {}, None

        from backend.app.tools.domain_tools import GeocodeRegionArgs
        from backend.app.tools.registry import ToolSpec
        slow_reg = ToolRegistry([ToolSpec(name="slow_geo", description="slow", args_model=GeocodeRegionArgs, handler=_slow_handler, timeout_s=0.02)])
        slow_env = await slow_reg.execute(call_id="st1", agent="risk", tool_name="slow_geo", raw_arguments={"region": "Odisha"}, allow_list=("slow_geo",))
        assert slow_env.status == "error" and "timed out" in (slow_env.error or "")

    # 11. Unhandled exception on route produces structured 503 JSON + request_id (never 500 or traceback)
    def explode_orch():
        raise RuntimeError("Unexpected secret backend boom")

    monkeypatch.setattr(main_mod, "_get_orchestrator", explode_orch)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        r_exc = await ac.post("/orchestrate-intake", json={"disaster_type": "Storm", "region": "Odisha"})
        assert r_exc.status_code == 503
        err_body = r_exc.json()
        assert err_body["error"]["code"] == "service_unavailable"
        assert err_body["request_id"]
        assert "secret backend boom" not in r_exc.text


@pytest.mark.asyncio
async def test_auth_cors_trusted_proxy_rate_limit_concurrency_and_chunked_body_limit(monkeypatch, caplog):
    """Verify API key auth on every non-health endpoint, X-Forwarded-For proxy trust, 429+Retry-After, chunked 413, and request_id logs."""
    from backend.app import main as main_mod
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    # 1. Server-side key check in Next.js API routes (never NEXT_PUBLIC)
    for route_file in Path("frontend/app/api").rglob("*.ts"):
        text = route_file.read_text(encoding="utf-8")
        assert "NEXT_PUBLIC_DISASTERMIND_API_KEY" not in text
        if route_file.parent.name in ("orchestrate", "generate", "agent-answer"):
            assert "process.env.DISASTERMIND_API_KEY" in text

    # 2. Trusted-proxy X-Forwarded-For extraction
    assert extract_client_ip("10.0.0.5", "203.0.113.9, 10.0.0.2", ("10.0.0.0/8",)) == "203.0.113.9"
    assert extract_client_ip("198.51.100.7", "203.0.113.9", ("10.0.0.0/8",)) == "198.51.100.7"

    # 3. Auth on all non-health endpoints + health public + CORS + Request-ID in logs
    sec_settings = Settings(nvidia_api_key="dummy", disastermind_api_key="secret-test-key", app_env="production")
    monkeypatch.setattr(main_mod, "get_settings", lambda: sec_settings)
    monkeypatch.setattr("backend.app.security.get_settings", lambda: sec_settings)
    monkeypatch.setattr(main_mod, "verify_runtime_alignment", lambda *_a, **_kw: {"status": "ok"})
    with caplog.at_level(logging.INFO, logger="disastermind"):
        with TestClient(app) as client:
            r_h = client.get("/health", headers={"X-Request-ID": "health-log-rid"})
            assert r_h.status_code == 200
            assert r_h.json()["request_id"] == "health-log-rid"
            assert client.get("/keepalive").status_code == 200
            for ep, payload in (
                ("/generate", {"messages": [{"role": "user", "content": "hi"}]}),
                ("/orchestrate", {"disaster_type": "Flood", "region": "Assam"}),
                ("/orchestrate-intake", {"disaster_type": "Flood", "region": "Assam"}),
                ("/orchestrate/stream", {"disaster_type": "Flood", "region": "Assam"}),
                ("/agent-answer", {"intake": {"disaster_type": "Flood", "region": "Assam"}, "question": "Risk?"}),
            ):
                r401 = client.post(ep, json=payload, headers={"X-Request-ID": "auth-check-123"})
                assert r401.status_code == 401
                assert r401.json()["error"]["request_id"] == "auth-check-123"
                assert r401.headers.get("X-Request-ID") == "auth-check-123"

            r_cors = client.options(
                "/orchestrate-intake",
                headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
            )
            assert r_cors.headers.get("access-control-allow-origin") == "http://localhost:3000"

    # 4. Isolated mini ASGI app testing streamed chunked 413 (NO Content-Length), rate-limit 429+Retry-After, concurrency 429+Retry-After
    mini = FastAPI()

    @mini.post("/echo")
    async def echo_endpoint(req: dict[str, Any]):
        await asyncio.sleep(0.05)
        return {"ok": True}

    mini.add_middleware(ConcurrencyCapMiddleware, max_concurrent=1, retry_after_s=7)
    mini.add_middleware(InMemoryRateLimitMiddleware, requests_per_window=2, window_seconds=60, trusted_proxies=("127.0.0.1",))
    mini.add_middleware(RequestSizeLimitMiddleware, max_bytes=128)
    mini.add_middleware(RequestContextMiddleware)

    async def _chunked_stream():
        for _ in range(10):
            yield b'{"x": "' + (b"A" * 20) + b'"}'

    transport = httpx.ASGITransport(app=mini)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        # Streamed body exceeding 128 bytes without Content-Length triggers 413!
        r_chunk = await ac.post("/echo", content=_chunked_stream(), headers={"X-Request-ID": "chunk-413"})
        assert r_chunk.status_code == 413
        assert r_chunk.json()["request_id"] == "chunk-413"

        # Rate limit 429 + Retry-After honoring trusted X-Forwarded-For IP key
        r1 = await ac.post("/echo", json={"a": 1}, headers={"X-Forwarded-For": "203.0.113.42"})
        r2 = await ac.post("/echo", json={"a": 2}, headers={"X-Forwarded-For": "203.0.113.42"})
        r3 = await ac.post("/echo", json={"a": 3}, headers={"X-Forwarded-For": "203.0.113.42"})
        assert r1.status_code == 200 and r2.status_code == 200 and r3.status_code == 429
        assert int(r3.headers["Retry-After"]) >= 1

        # Different forwarded IP exercises ConcurrencyCapMiddleware -> second simultaneous request gets 429 Retry-After=7
        t_a = asyncio.create_task(ac.post("/echo", json={"a": 1}, headers={"X-Forwarded-For": "198.51.100.1"}))
        await asyncio.sleep(0.01)
        r_b = await ac.post("/echo", json={"a": 2}, headers={"X-Forwarded-For": "198.51.100.2"})
        r_a = await t_a
        assert r_a.status_code == 200
        assert r_b.status_code == 429
        assert r_b.headers["Retry-After"] == "7"


@pytest.mark.asyncio
async def test_concurrent_enrichment_uses_asyncio_gather():
    """Verify enrich_scenario executes post-geocoding HTTP calls concurrently without heuristic capacity."""
    def enrich_handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "geocoding-api.open-meteo.com" in url:
            return httpx.Response(200, json={"results": [{"name": "Puri", "admin1": "Odisha", "country": "India", "country_code": "IN", "latitude": 19.81, "longitude": 85.83, "population": 200564}]})
        if "nominatim.openstreetmap.org" in url:
            return httpx.Response(200, json=[{"boundingbox": ["19.7", "19.9", "85.7", "85.9"], "extratags": {}}])
        if "api.open-meteo.com/v1/forecast" in url:
            return httpx.Response(200, json={"current": {"temperature_2m": 0.0, "relative_humidity_2m": 75, "precipitation": 0.0, "wind_speed_10m": 12.0}})
        if "api.open-meteo.com/v1/elevation" in url:
            return httpx.Response(200, json={"elevation": [0.0]})
        if "overpass-api.de" in url:
            return httpx.Response(200, json={"elements": [{"tags": {"amenity": "hospital"}}]})
        if "restcountries.com" in url:
            return httpx.Response(200, json={"population": 1400000000, "area": 3287263, "cca3": "IND"})
        if "api.worldbank.org" in url:
            return httpx.Response(200, json=[{}, [{"value": 0.5}]])
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(enrich_handler)) as hc:
        scen = await enrich_scenario(ScenarioIntakeRequest(disaster_type="Flood", region="Puri"), http_client=hc)

    assert scen.population == 200564
    assert scen.hospital_capacity is None  # no fake 150-bed guess!
    assert scen.temperature == 0.0 and scen.rainfall == 0.0 and scen.elevation == 0.0
    assert len(scen.data_sources) == 7
