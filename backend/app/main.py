from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import get_settings
from backend.app.schemas import (
    AgentQuestionRequest,
    AgentQuestionResponse,
    DisasterScenario,
    GenerationRequest,
    GenerationResponse,
    OrchestrationResponse,
    ScenarioIntakeRequest,
)
from backend.app.security import (
    InMemoryRateLimitMiddleware,
    RequestSizeLimitMiddleware,
    SecurityHeadersMiddleware,
    require_api_key,
)
from backend.app.services.orchestration import DisasterMindOrchestrator
from backend.app.services.nvidia_generation import (
    NvidiaBuildGenerationClient,
    NvidiaGenerationConfigError,
    NvidiaGenerationError,
)

settings = get_settings()
app = FastAPI(
    title="DisasterMind AI Backend",
    description="Multi-agent disaster orchestration backed by NVIDIA Build / NIM generation.",
    version="0.1.0",
    docs_url=None if settings.production else "/docs",
    redoc_url=None if settings.production else "/redoc",
    openapi_url=None if settings.production else "/openapi.json",
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


@app.get("/health")
async def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "nvidia_configured": bool(settings.nvidia_api_key),
        "nvidia_model": settings.nvidia_model,
        "api_key_required": settings.api_key_required,
        "environment": settings.app_env,
    }


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
    response_model=OrchestrationResponse,
    dependencies=[Depends(require_api_key)],
)
async def orchestrate(request: DisasterScenario) -> OrchestrationResponse:
    orchestrator = DisasterMindOrchestrator()
    return await orchestrator.run(request)


@app.post(
    "/orchestrate-intake",
    response_model=OrchestrationResponse,
    dependencies=[Depends(require_api_key)],
)
async def orchestrate_intake(request: ScenarioIntakeRequest) -> OrchestrationResponse:
    orchestrator = DisasterMindOrchestrator()
    return await orchestrator.run_from_intake(request)


@app.post(
    "/agent-answer",
    response_model=AgentQuestionResponse,
    dependencies=[Depends(require_api_key)],
)
async def agent_answer(request: AgentQuestionRequest) -> AgentQuestionResponse:
    orchestrator = DisasterMindOrchestrator()
    if request.intake is not None:
        return await orchestrator.answer_question_from_intake(request.intake, request.question)
    if request.scenario is None:
        raise HTTPException(status_code=422, detail="Either scenario or intake is required.")
    return await orchestrator.answer_question(request.scenario, request.question)
