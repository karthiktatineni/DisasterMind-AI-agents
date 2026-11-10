"""FastAPI entrypoint for DisasterMind AI 4-agent orchestration and SSE streaming."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import httpx

from backend.app.config import get_settings
from backend.app.orchestration.models import OrchestrationRunResult, SSEEventRecord
from backend.app.orchestration.orchestrator import DisasterMindOrchestrator
from backend.app.schemas import (
    AgentQuestionRequest,
    AgentQuestionResponse,
    DisasterScenario,
    GenerationRequest,
    GenerationResponse,
    ScenarioIntakeRequest,
)
from backend.app.security import (
    InMemoryRateLimitMiddleware,
    RequestSizeLimitMiddleware,
    SecurityHeadersMiddleware,
    require_api_key,
)
from backend.app.services.keepalive import KeepAliveService
from backend.app.services.llm_client import LLMClientConfigError, NativeToolLLMClient
from backend.app.services.nvidia_generation import (
    NvidiaBuildGenerationClient,
    NvidiaGenerationConfigError,
    NvidiaGenerationError,
)
from backend.app.services.snapshot_guard import verify_runtime_alignment

settings = get_settings()
keepalive_service = KeepAliveService(settings)


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    cfg = get_settings()
    verify_runtime_alignment(cfg)
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(90.0, connect=20.0),
        limits=httpx.Limits(max_connections=30, max_keepalive_connections=10),
        headers={"User-Agent": "DisasterMindAI/0.2"},
    )
    app_instance.state.http_client = http_client
    try:
        app_instance.state.llm_client = NativeToolLLMClient(settings=cfg, http_client=http_client)
    except LLMClientConfigError:
        app_instance.state.llm_client = None
    await keepalive_service.start()
    try:
        yield
    finally:
        await keepalive_service.stop()
        await http_client.aclose()


app = FastAPI(
    title="DisasterMind AI Backend",
    description="4-agent native tool-calling disaster briefing orchestration backed by NVIDIA NIM & Pinecone.",
    version="0.2.0",
    docs_url=None if settings.production else "/docs",
    redoc_url=None if settings.production else "/redoc",
    openapi_url=None if settings.production else "/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-DisasterMind-API-Key", "X-Request-ID"],
)
app.add_middleware(
    InMemoryRateLimitMiddleware,
    requests_per_window=settings.rate_limit_requests,
    window_seconds=settings.rate_limit_window_seconds,
)
app.add_middleware(RequestSizeLimitMiddleware, max_bytes=settings.max_request_bytes)
app.add_middleware(SecurityHeadersMiddleware, settings=settings)


def _get_orchestrator() -> DisasterMindOrchestrator:
    http_client = getattr(app.state, "http_client", None)
    llm_client = getattr(app.state, "llm_client", None)
    return DisasterMindOrchestrator(
        settings=get_settings(),
        llm_client=llm_client,
        http_client=http_client,
    )


@app.get("/health")
async def health() -> dict[str, str | bool | int]:
    cfg = get_settings()
    return {
        "status": "ok",
        "nvidia_configured": bool(cfg.nvidia_api_key),
        "nvidia_model": cfg.nvidia_model,
        "api_key_required": cfg.api_key_required,
        "environment": cfg.app_env,
        "backend_keepalive_configured": keepalive_service.backend_enabled,
        "pinecone_configured": bool(cfg.pinecone_api_key and cfg.pinecone_index_name),
    }


@app.get("/keepalive")
async def keepalive() -> dict:
    return {"status": "ok", **keepalive_service.status()}


@app.post(
    "/generate",
    response_model=GenerationResponse,
    dependencies=[Depends(require_api_key)],
)
async def generate(request: GenerationRequest) -> GenerationResponse:
    try:
        client = NvidiaBuildGenerationClient()
        return await client.generate(
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
    except NvidiaGenerationConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except NvidiaGenerationError as exc:
        raise HTTPException(
            status_code=502,
            detail="NVIDIA generation failed. Check backend service configuration.",
        ) from exc


@app.post(
    "/orchestrate",
    response_model=OrchestrationRunResult,
    dependencies=[Depends(require_api_key)],
)
async def orchestrate(request: DisasterScenario) -> OrchestrationRunResult:
    orchestrator = _get_orchestrator()
    return await orchestrator.run(request)


@app.post(
    "/orchestrate-intake",
    response_model=OrchestrationRunResult,
    dependencies=[Depends(require_api_key)],
)
async def orchestrate_intake(request: ScenarioIntakeRequest) -> OrchestrationRunResult:
    orchestrator = _get_orchestrator()
    return await orchestrator.run(request)


@app.post(
    "/orchestrate/stream",
    dependencies=[Depends(require_api_key)],
)
async def orchestrate_stream(request: ScenarioIntakeRequest) -> StreamingResponse:
    orchestrator = _get_orchestrator()
    queue: asyncio.Queue[SSEEventRecord | None] = asyncio.Queue()

    def _push_event(ev: SSEEventRecord) -> None:
        queue.put_nowait(ev)

    async def _run_job() -> None:
        try:
            await orchestrator.run(request, on_event=_push_event)
        except Exception as exc:
            queue.put_nowait(
                SSEEventRecord(
                    event="run_failed",
                    run_id="unknown",
                    data={"error": str(exc)},
                )
            )
        finally:
            queue.put_nowait(None)

    async def _event_generator():
        task = asyncio.create_task(_run_job())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                payload = json.dumps(item.model_dump(), ensure_ascii=False)
                yield f"event: {item.event}\ndata: {payload}\n\n"
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(_event_generator(), media_type="text/event-stream")


@app.post(
    "/agent-answer",
    response_model=AgentQuestionResponse,
    dependencies=[Depends(require_api_key)],
)
async def agent_answer(request: AgentQuestionRequest) -> AgentQuestionResponse:
    orchestrator = _get_orchestrator()
    target = request.intake if request.intake is not None else request.scenario
    if target is None:
        raise HTTPException(status_code=422, detail="Either scenario or intake is required.")
    res = await orchestrator.run(target)
    b = res.briefing
    return AgentQuestionResponse(
        request_id=b.run_id,
        answer=b.executive_summary or f"Status: {b.status}",
        provider="nvidia-nim" if b.model_ids else "none",
        model=b.model_ids[0] if b.model_ids else None,
        cited_agents=[k for k in ("risk", "logistics", "evacuation") if getattr(b, "status") != "insufficient_data"],
        confidence=1.0 if b.status == "complete" else (0.5 if b.status == "partial" else 0.0),
    )
