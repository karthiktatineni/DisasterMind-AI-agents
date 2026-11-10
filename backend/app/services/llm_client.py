"""Async NVIDIA NIM tool-calling LLM client with RPM limiting, jittered retries, and strict 410 handling."""

from __future__ import annotations

import asyncio
from collections import deque
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
)

from backend.app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class LLMClientConfigError(RuntimeError):
    """Raised when required LLM credentials or model configuration are invalid."""


class LLMClientError(RuntimeError):
    """Raised on upstream LLM errors (410 deprecated, 401/403 auth, exhausted retries, timeout)."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class ToolCallRequest:
    id: str
    name: str
    raw_arguments: str


@dataclass(frozen=True)
class LLMToolResponse:
    model: str
    request_id: str | None
    finish_reason: str | None
    content: str | None
    reasoning_content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0


class _SlidingWindowRPMLimiter:
    """Bounds request rate below NIM's measured ~25-40 RPM prototyping ceiling."""

    def __init__(self, max_rpm: int = 30, max_concurrency: int = 4) -> None:
        self.max_rpm = max(1, int(max_rpm))
        self._semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> asyncio.Semaphore:
        await self._semaphore.acquire()
        try:
            while True:
                wait_for = 0.0
                async with self._lock:
                    now = time.monotonic()
                    cutoff = now - 60.0
                    while self._timestamps and self._timestamps[0] < cutoff:
                        self._timestamps.popleft()
                    if len(self._timestamps) < self.max_rpm:
                        self._timestamps.append(now)
                        return self._semaphore
                    wait_for = max(0.05, (self._timestamps[0] + 60.05) - now)
                await asyncio.sleep(wait_for)
        except Exception:
            self._semaphore.release()
            raise

    def release(self) -> None:
        self._semaphore.release()


class NativeToolLLMClient:
    """Shared async OpenAI-compatible client enforcing native tools + retry/backoff."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        http_client: httpx.AsyncClient | None = None,
        openai_client: AsyncOpenAI | None = None,
        max_rpm: int = 18,
        max_concurrency: int = 2,
    ) -> None:
        self.settings = settings or get_settings()
        self._owns_http = http_client is None and openai_client is None
        self._http_client = http_client
        self._limiter = _SlidingWindowRPMLimiter(max_rpm=max_rpm, max_concurrency=max_concurrency)

        if openai_client is not None:
            self._client = openai_client
            return

        if not self.settings.nvidia_api_key:
            raise LLMClientConfigError("NVIDIA_API_KEY is missing. Configure .env before live LLM calls.")

        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(90.0, connect=20.0),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )

        self._client = AsyncOpenAI(
            api_key=self.settings.nvidia_api_key,
            base_url=self.settings.nvidia_base_url,
            http_client=self._http_client,
            timeout=90.0,
            max_retries=0,  # retries handled explicitly here with 410 fail-fast & jitter
        )

    async def aclose(self) -> None:
        if self._owns_http and self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

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
        max_retries: int = 5,
        base_delay: float = 3.0,
        request_timeout_s: float = 75.0,
    ) -> LLMToolResponse:
        if not tools:
            raise ValueError("Task 4 control flow requires every LLM request to include a non-empty tools list.")

        target_model = model or self.settings.nvidia_model
        last_exc: Exception | None = None

        for attempt in range(max_retries + 1):
            await self._limiter.acquire()
            started = time.perf_counter()
            try:
                kwargs: dict[str, Any] = {
                    "model": target_model,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": tool_choice,
                    "temperature": temperature,
                    "seed": seed,
                    "max_tokens": max_tokens,
                    "stream": False,
                }
                if isinstance(tool_choice, str) and len(tools) > 1:
                    kwargs["parallel_tool_calls"] = True
                completion = await asyncio.wait_for(
                    self._client.chat.completions.create(**kwargs),
                    timeout=request_timeout_s,
                )
                elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
                return self._parse_completion(completion, target_model, elapsed_ms)
            except APIStatusError as exc:
                status = exc.status_code
                last_exc = exc
                if status in (401, 403, 404, 410, 422):
                    raise LLMClientError(
                        f"NVIDIA LLM non-retryable HTTP {status} for model '{target_model}': {exc}",
                        status_code=status,
                    ) from exc
                if status not in (408, 429) and status < 500:
                    raise LLMClientError(f"NVIDIA LLM HTTP {status}: {exc}", status_code=status) from exc
                if attempt >= max_retries:
                    raise LLMClientError(
                        f"NVIDIA LLM HTTP {status} after {attempt + 1} attempts: {exc}",
                        status_code=status,
                    ) from exc
            except (APITimeoutError, asyncio.TimeoutError) as exc:
                last_exc = exc
                if attempt >= max_retries:
                    raise LLMClientError(f"NVIDIA LLM timed out after {attempt + 1} attempts: {exc}") from exc
            except (APIConnectionError, APIError) as exc:
                last_exc = exc
                if attempt >= max_retries:
                    raise LLMClientError(f"NVIDIA LLM error after {attempt + 1} attempts: {exc}") from exc
            finally:
                self._limiter.release()

            jitter = random.uniform(0.0, 0.5 * base_delay)
            delay = min(30.0, base_delay * (2**attempt) + jitter)
            logger.warning("Retrying LLM request (%s) in %.2fs after %s", target_model, delay, type(last_exc).__name__)
            await asyncio.sleep(delay)

        raise LLMClientError(f"NVIDIA LLM failed unexpectedly: {last_exc}")

    @staticmethod
    def _parse_completion(completion: Any, fallback_model: str, elapsed_ms: float) -> LLMToolResponse:
        choices = getattr(completion, "choices", None) or []
        if not choices:
            raise LLMClientError("NVIDIA completion contained empty choices.")
        choice = choices[0]
        msg = choice.message
        raw_calls = getattr(msg, "tool_calls", None) or []
        parsed_calls: list[ToolCallRequest] = []
        for idx, tc in enumerate(raw_calls):
            fn = getattr(tc, "function", None)
            if fn is None:
                continue
            parsed_calls.append(
                ToolCallRequest(
                    id=str(getattr(tc, "id", None) or f"call-{idx}"),
                    name=str(getattr(fn, "name", "")),
                    raw_arguments=str(getattr(fn, "arguments", "") or "{}"),
                )
            )
        usage = getattr(completion, "usage", None)
        comp_details = getattr(usage, "completion_tokens_details", None) if usage else None
        return LLMToolResponse(
            model=str(getattr(completion, "model", None) or fallback_model),
            request_id=getattr(completion, "id", None),
            finish_reason=getattr(choice, "finish_reason", None),
            content=getattr(msg, "content", None),
            reasoning_content=getattr(msg, "reasoning_content", None),
            tool_calls=parsed_calls,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0,
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0,
            reasoning_tokens=int(getattr(comp_details, "reasoning_tokens", 0) or 0) if comp_details else 0,
            total_tokens=int(getattr(usage, "total_tokens", 0) or 0) if usage else 0,
            latency_ms=elapsed_ms,
        )

