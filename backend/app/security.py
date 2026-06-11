from __future__ import annotations

import hmac
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable

from fastapi import HTTPException, Request, Response, status
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from backend.app.config import Settings, get_settings


PUBLIC_PATHS = {"/health", "/keepalive", "/docs", "/openapi.json", "/redoc"}


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_bytes: int) -> None:
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_bytes:
            return JSONResponse(
                {"detail": "Request body exceeds configured size limit."},
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            )
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, settings: Settings) -> None:
        super().__init__(app)
        self.settings = settings

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-site"
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        if self.settings.production:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


class InMemoryRateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_window: int, window_seconds: int) -> None:
        super().__init__(app)
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.client_windows: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        client_host = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window = self.client_windows[client_host]
        while window and now - window[0] > self.window_seconds:
            window.popleft()

        if len(window) >= self.requests_per_window:
            return JSONResponse(
                {"detail": "Rate limit exceeded."},
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            )

        window.append(now)
        return await call_next(request)


async def require_api_key(request: Request) -> None:
    settings = get_settings()
    if not settings.api_key_required:
        return

    if settings.production and not settings.disastermind_api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="DISASTERMIND_API_KEY must be configured in production.",
        )

    supplied_key = request.headers.get("X-DisasterMind-API-Key", "")
    if not supplied_key or not hmac.compare_digest(
        supplied_key,
        settings.disastermind_api_key,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
