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
from backend.app.services.keepalive import KeepAliveService
from backend.app.services.orchestration import DisasterMindOrchestrator
from backend.app.schemas import AgentResult

def _normalize_agent_results(results: list[AgentResult]) -> None:
    for res in results:
        if res.agent_name == "validation":
            res.output.setdefault("issues", [])
            res.output.setdefault("warnings", [])
            res.output.setdefault("validation_status", "failed")
        elif res.agent_name == "planner":
            res.output.setdefault("phases", [])
            res.output.setdefault("evacuation_strategy", "Unknown")
            res.output.setdefault("resource_estimates", {})
            res.output.setdefault("shelter_assignments", {})
        elif res.agent_name == "logistics":
            res.output.setdefault("resource_plan", {})
            res.output.setdefault("resource_shortfalls", [])
        elif res.agent_name == "evacuation":
            res.output.setdefault("evacuation_zones", [])
            res.output.setdefault("shelter_gap", 0)
            res.output.setdefault("route_conflicts", [])
        elif res.agent_name == "simulation":
            res.output.setdefault("scenario_results", [])
            res.output.setdefault("resilience_gaps", [])
        elif res.agent_name == "decision":
            res.output.setdefault("executive_recommendation", "Awaiting decision.")
            res.output.setdefault("prioritized_actions", [])
            res.output.setdefault("unresolved_risks", [])
        elif res.agent_name == "explainability":
            res.output.setdefault("explanations", {})
        elif res.agent_name == "knowledge":
            res.output.setdefault("historical_events", [])
            res.output.setdefault("damage_patterns", {})
        elif res.agent_name == "reporting":
            res.output.setdefault("sections", [])
            res.output.setdefault("report_text", "")
            res.output.setdefault("evidence_used", [])
from backend.app.services.nvidia_generation import (
    NvidiaBuildGenerationClient,
    NvidiaGenerationConfigError,
    NvidiaGenerationError,
)

settings = get_settings()
keepalive_service = KeepAliveService(settings)
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


@app.on_event("startup")
async def start_keepalive() -> None:
    await keepalive_service.start()


@app.on_event("shutdown")
async def stop_keepalive() -> None:
    await keepalive_service.stop()


@app.get("/health")
async def health() -> dict[str, str | bool | int]:
    return {
        "status": "ok",
        "nvidia_configured": bool(settings.nvidia_api_key),
        "nvidia_model": settings.nvidia_model,
        "api_key_required": settings.api_key_required,
        "environment": settings.app_env,
        "backend_keepalive_configured": keepalive_service.backend_enabled,
        "supabase_keepalive_configured": keepalive_service.supabase_enabled,
    }


@app.get("/keepalive")
async def keepalive() -> dict:
    return {
        "status": "ok",
        **keepalive_service.status(),
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
    response = await orchestrator.run(request)
    _normalize_agent_results(response.agent_results)
    return response


@app.post(
    "/orchestrate-intake",
    response_model=OrchestrationResponse,
    dependencies=[Depends(require_api_key)],
)
async def orchestrate_intake(request: ScenarioIntakeRequest) -> OrchestrationResponse:
    orchestrator = DisasterMindOrchestrator()
    response = await orchestrator.run_from_intake(request)
    _normalize_agent_results(response.agent_results)
    return response


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
