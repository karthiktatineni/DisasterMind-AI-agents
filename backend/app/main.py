"""FastAPI entrypoint for DisasterMind AI 4-agent orchestration and SSE streaming (Task 7 hardened)."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import logging

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
from openai import AsyncOpenAI
from starlette.exceptions import HTTPException as StarletteHTTPException

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
    ConcurrencyCapMiddleware,
    InMemoryRateLimitMiddleware,
    RequestContextMiddleware,
    RequestSizeLimitMiddleware,
    SecurityHeadersMiddleware,
    error_envelope,
    get_request_id,
    install_request_id_logging,
    require_api_key,
)
from backend.app.services.historical_stats import HistoricalStatsService
from backend.app.services.keepalive import KeepAliveService
from backend.app.services.llm_client import LLMClientConfigError, NativeToolLLMClient
from backend.app.services.nvidia_generation import (
    NvidiaBuildGenerationClient,
    NvidiaGenerationConfigError,
    NvidiaGenerationError,
)
from backend.app.services.retrieval_service import RetrievalService
from backend.app.services.snapshot_guard import verify_runtime_alignment

logger = logging.getLogger("disastermind")
settings = get_settings()
keepalive_service = KeepAliveService(settings)


def _load_stats_optional() -> HistoricalStatsService | None:
    try:
        return HistoricalStatsService()
    except FileNotFoundError:
        return None


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    install_request_id_logging()
    cfg = get_settings()
    await asyncio.to_thread(verify_runtime_alignment, cfg)
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(cfg.http_timeout_s, connect=20.0),
        limits=httpx.Limits(max_connections=30, max_keepalive_connections=10),
        headers={"User-Agent": "DisasterMindAI/0.2"},
    )
    app_instance.state.http_client = http_client
    app_instance.state.async_openai = (
        AsyncOpenAI(
            api_key=cfg.nvidia_api_key,
            base_url=cfg.nvidia_base_url,
            http_client=http_client,
            timeout=cfg.http_timeout_s,
            max_retries=0,
        )
        if cfg.nvidia_api_key
        else None
    )
    try:
        app_instance.state.llm_client = (
            NativeToolLLMClient(
                settings=cfg,
                http_client=http_client,
                openai_client=app_instance.state.async_openai,
            )
            if app_instance.state.async_openai is not None
            else None
        )
    except LLMClientConfigError:
        app_instance.state.llm_client = None

    app_instance.state.historical_stats_service = await asyncio.to_thread(_load_stats_optional)
    app_instance.state.retrieval_service = RetrievalService(settings=cfg)
    keepalive_service._http_client = http_client
    await keepalive_service.start()
    try:
        yield
    finally:
        await keepalive_service.stop()
        rs = getattr(app_instance.state, "retrieval_service", None)
        if rs is not None:
            if getattr(rs, "_embedding_service", None) is not None:
                await rs._embedding_service.aclose()
            if getattr(rs, "_pinecone_client", None) is not None:
                await rs._pinecone_client.aclose()
        if app_instance.state.async_openai is not None:
            await app_instance.state.async_openai.close()
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


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    rid = get_request_id(request)
    code_status = exc.status_code if exc.status_code != 500 else status.HTTP_503_SERVICE_UNAVAILABLE
    msg = exc.detail if isinstance(exc.detail, str) else "Request failed."
    return JSONResponse(
        status_code=code_status,
        content=error_envelope(code_status, code=f"http_{code_status}", message=msg, request_id=rid),
        headers={"X-Request-ID": rid},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    rid = get_request_id(request)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={
            **error_envelope(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                code="validation_error",
                message="Request payload failed validation.",
                request_id=rid,
            ),
            "validation_errors": exc.errors(),
        },
        headers={"X-Request-ID": rid},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    rid = get_request_id(request)
    logger.error("Unhandled request error (%s): %s", rid, type(exc).__name__)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=error_envelope(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            code="service_unavailable",
            message="Service encountered a transient upstream or internal failure.",
            request_id=rid,
        ),
        headers={"X-Request-ID": rid},
    )


# Middleware order (last added = outermost): RequestContext -> CORS -> SecurityHeaders -> SizeLimit -> RateLimit -> ConcurrencyCap
app.add_middleware(
    ConcurrencyCapMiddleware,
    max_concurrent=None,
    retry_after_s=15,
)
app.add_middleware(
    InMemoryRateLimitMiddleware,
    requests_per_window=settings.rate_limit_requests,
    window_seconds=settings.rate_limit_window_seconds,
    trusted_proxies=settings.trusted_proxies,
)
app.add_middleware(RequestSizeLimitMiddleware, max_bytes=settings.max_request_bytes)
app.add_middleware(SecurityHeadersMiddleware, settings=settings)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-DisasterMind-API-Key", "X-Request-ID"],
)
app.add_middleware(RequestContextMiddleware)


def _get_orchestrator() -> DisasterMindOrchestrator:
    http_client = getattr(app.state, "http_client", None)
    llm_client = getattr(app.state, "llm_client", None)
    retrieval_service = getattr(app.state, "retrieval_service", None)
    historical_stats_service = getattr(app.state, "historical_stats_service", None)
    return DisasterMindOrchestrator(
        settings=get_settings(),
        llm_client=llm_client,
        retrieval_service=retrieval_service,
        historical_stats_service=historical_stats_service,
        http_client=http_client,
    )


@app.get("/health")
async def health(request: Request) -> dict[str, str | bool | int]:
    cfg = get_settings()
    return {
        "status": "ok",
        "request_id": get_request_id(request),
        "nvidia_configured": bool(cfg.nvidia_api_key),
        "nvidia_model": cfg.nvidia_model,
        "api_key_required": cfg.api_key_required,
        "environment": cfg.app_env,
        "backend_keepalive_configured": keepalive_service.backend_enabled,
        "pinecone_configured": bool(cfg.pinecone_api_key and cfg.pinecone_index_name),
        "max_concurrent_orchestrations": cfg.max_concurrent_orchestrations,
    }


@app.get("/keepalive")
async def keepalive(request: Request) -> dict:
    return {"status": "ok", "request_id": get_request_id(request), **keepalive_service.status()}


@app.post(
    "/generate",
    response_model=GenerationResponse,
    dependencies=[Depends(require_api_key)],
)
async def generate(request: GenerationRequest, http_request: Request) -> GenerationResponse:
    rid = get_request_id(http_request)
    try:
        shared_openai = getattr(app.state, "async_openai", None)
        client = NvidiaBuildGenerationClient(settings=get_settings(), client=shared_openai)
        logger.info("Handling /generate request_id=%s", rid)
        return await client.generate(
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
    except NvidiaGenerationConfigError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except NvidiaGenerationError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="NVIDIA generation failed. Check backend service configuration.",
        ) from exc


@app.post(
    "/orchestrate",
    response_model=OrchestrationRunResult,
    dependencies=[Depends(require_api_key)],
)
async def orchestrate(request: DisasterScenario, http_request: Request) -> OrchestrationRunResult:
    rid = get_request_id(http_request)
    logger.info("Handling /orchestrate request_id=%s", rid)
    orchestrator = _get_orchestrator()
    return await orchestrator.run(request, run_id=rid)


@app.post(
    "/orchestrate-intake",
    response_model=OrchestrationRunResult,
    dependencies=[Depends(require_api_key)],
)
async def orchestrate_intake(request: ScenarioIntakeRequest, http_request: Request) -> OrchestrationRunResult:
    rid = get_request_id(http_request)
    logger.info("Handling /orchestrate-intake request_id=%s", rid)
    orchestrator = _get_orchestrator()
    return await orchestrator.run(request, run_id=rid)


@app.post(
    "/orchestrate/stream",
    dependencies=[Depends(require_api_key)],
)
async def orchestrate_stream(request: ScenarioIntakeRequest, http_request: Request) -> StreamingResponse:
    rid = get_request_id(http_request)
    logger.info("Handling /orchestrate/stream request_id=%s", rid)
    orchestrator = _get_orchestrator()
    queue: asyncio.Queue[SSEEventRecord | None] = asyncio.Queue()

    def _push_event(ev: SSEEventRecord) -> None:
        queue.put_nowait(ev)

    async def _run_job() -> None:
        try:
            await orchestrator.run(request, run_id=rid, on_event=_push_event)
        except Exception as exc:
            logger.error("SSE orchestration failure request_id=%s err=%s", rid, type(exc).__name__)
            queue.put_nowait(
                SSEEventRecord(
                    event="run_failed",
                    run_id=rid,
                    data={"error": "Orchestration encountered an upstream failure.", "request_id": rid},
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

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"X-Request-ID": rid, "Cache-Control": "no-cache"},
    )


@app.post(
    "/agent-answer",
    response_model=AgentQuestionResponse,
    dependencies=[Depends(require_api_key)],
)
async def agent_answer(request: AgentQuestionRequest, http_request: Request) -> AgentQuestionResponse:
    rid = get_request_id(http_request)
    logger.info("Handling /agent-answer request_id=%s", rid)
    orchestrator = _get_orchestrator()
    target = request.intake if request.intake is not None else request.scenario
    if target is None:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Either scenario or intake is required.")
    res = await orchestrator.run(target, run_id=rid)
    b = res.briefing
    return AgentQuestionResponse(
        request_id=rid,
        answer=b.executive_summary or f"Status: {b.status}",
        provider="nvidia-nim" if b.model_ids else "none",
        model=b.model_ids[0] if b.model_ids else None,
        cited_agents=[k for k in ("risk", "logistics", "evacuation") if b.status != "insufficient_data"],
        confidence=1.0 if b.status == "complete" else (0.5 if b.status == "partial" else 0.0),
    )
