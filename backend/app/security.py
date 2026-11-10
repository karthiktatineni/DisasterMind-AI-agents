"""Async security, request-ID logging/envelopes, streamed body-size limit, trusted-proxy IP rate limiting, and concurrency gate (Task 7)."""

from __future__ import annotations

import asyncio
import contextvars
import hmac
import ipaddress
import json
import logging
import math
import re
import time
from collections import defaultdict, deque
from collections.abc import Iterable
from typing import Any
from uuid import uuid4

from fastapi import HTTPException, Request, status
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from backend.app.config import Settings, get_settings

PUBLIC_PATHS = {"/health", "/keepalive", "/docs", "/openapi.json", "/redoc"}
REQUEST_ID_CTX: contextvars.ContextVar[str] = contextvars.ContextVar("disastermind_request_id", default="none")
_SAFE_REQ_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{4,64}$")


class RequestIdLogFilter(logging.Filter):
    """Attach active correlation request_id to every backend log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = REQUEST_ID_CTX.get()
        return True


def install_request_id_logging() -> None:
    flt = RequestIdLogFilter()
    for target in (logging.getLogger(), logging.getLogger("disastermind"), logging.getLogger("uvicorn.error")):
        if not any(isinstance(existing, RequestIdLogFilter) for existing in target.filters):
            target.addFilter(flt)


def get_request_id(request: Request | None = None) -> str:
    if request is not None:
        rid = getattr(getattr(request, "state", None), "request_id", None)
        if rid:
            return str(rid)
    ctx = REQUEST_ID_CTX.get()
    return ctx if ctx != "none" else str(uuid4())


def error_envelope(
    status_code: int,
    *,
    code: str,
    message: str,
    request_id: str,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "detail": message,
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id,
        },
    }


async def _send_json(
    send: Send,
    status_code: int,
    payload: dict[str, Any],
    *,
    request_id: str,
    extra_headers: list[tuple[bytes, bytes]] | None = None,
) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers: list[tuple[bytes, bytes]] = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
        (b"x-request-id", request_id.encode("ascii", errors="ignore")),
    ]
    if extra_headers:
        headers.extend(extra_headers)
    await send({"type": "http.response.start", "status": status_code, "headers": headers})
    await send({"type": "http.response.body", "body": body, "more_body": False})


def _sanitize_request_id(raw: str | None) -> str:
    if raw and _SAFE_REQ_ID_RE.match(raw.strip()):
        return raw.strip()
    return str(uuid4())


def _headers_dict(scope: Scope) -> dict[str, str]:
    return {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", ())}


def _parse_trusted_networks(entries: Iterable[str]) -> list[ipaddress._BaseNetwork]:
    nets: list[ipaddress._BaseNetwork] = []
    for item in entries:
        text = item.strip()
        if not text:
            continue
        try:
            nets.append(ipaddress.ip_network(text, strict=False))
        except ValueError:
            continue
    return nets


def _is_trusted_ip(ip_str: str, trusted_nets: list[ipaddress._BaseNetwork]) -> bool:
    if not trusted_nets or not ip_str:
        return False
    try:
        addr = ipaddress.ip_address(ip_str.strip())
    except ValueError:
        return False
    return any(addr in net for net in trusted_nets)


def extract_client_ip(
    peer_ip: str | None,
    forwarded_for_header: str | None,
    trusted_proxies: tuple[str, ...],
) -> str:
    """Resolve client IP from X-Forwarded-For ONLY when the direct peer is in trusted_proxies."""
    direct = (peer_ip or "unknown").strip() or "unknown"
    trusted_nets = _parse_trusted_networks(trusted_proxies)
    if not forwarded_for_header or not _is_trusted_ip(direct, trusted_nets):
        return direct
    hops = [h.strip() for h in forwarded_for_header.split(",") if h.strip()]
    for hop in reversed(hops):
        if not _is_trusted_ip(hop, trusted_nets):
            return hop
    return hops[0] if hops else direct


class RequestContextMiddleware:
    """Pure-ASGI correlation request_id middlewarepropagating X-Request-ID across logs, headers, and errors."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = _headers_dict(scope)
        rid = _sanitize_request_id(headers.get("x-request-id"))
        state = scope.setdefault("state", {})
        if isinstance(state, dict):
            state["request_id"] = rid
        token = REQUEST_ID_CTX.set(rid)

        async def _send_with_rid(message: Message) -> None:
            if message["type"] == "http.response.start":
                raw_headers = list(message.get("headers", ()))
                if not any(k.lower() == b"x-request-id" for k, _ in raw_headers):
                    raw_headers.append((b"x-request-id", rid.encode("ascii")))
                message = {**message, "headers": raw_headers}
            await send(message)

        try:
            await self.app(scope, receive, _send_with_rid)
        finally:
            REQUEST_ID_CTX.reset(token)


class RequestSizeLimitMiddleware:
    """Pure-ASGI limit enforcing BOTH Content-Length header AND streamed chunked bytes without buffering oversized payloads."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        rid = REQUEST_ID_CTX.get()
        if rid == "none":
            rid = str(uuid4())
        headers = _headers_dict(scope)
        cl_raw = headers.get("content-length")
        if cl_raw is not None:
            try:
                cl_val = int(cl_raw)
            except ValueError:
                await _send_json(
                    send,
                    status.HTTP_400_BAD_REQUEST,
                    error_envelope(
                        status.HTTP_400_BAD_REQUEST,
                        code="invalid_content_length",
                        message="Invalid Content-Length header.",
                        request_id=rid,
                    ),
                    request_id=rid,
                )
                return
            if cl_val > self.max_bytes:
                await _send_json(
                    send,
                    status.HTTP_413_CONTENT_TOO_LARGE,
                    error_envelope(
                        status.HTTP_413_CONTENT_TOO_LARGE,
                        code="payload_too_large",
                        message="Request body exceeds configured size limit.",
                        request_id=rid,
                    ),
                    request_id=rid,
                )
                return

        method = str(scope.get("method", "GET")).upper()
        if method in ("GET", "HEAD", "OPTIONS", "DELETE"):
            await self.app(scope, receive, send)
            return

        chunks: list[Message] = []
        bytes_seen = 0
        more_body = True
        while more_body:
            msg = await receive()
            if msg["type"] != "http.request":
                chunks.append(msg)
                break
            body = msg.get("body", b"")
            bytes_seen += len(body)
            if bytes_seen > self.max_bytes:
                await _send_json(
                    send,
                    status.HTTP_413_CONTENT_TOO_LARGE,
                    error_envelope(
                        status.HTTP_413_CONTENT_TOO_LARGE,
                        code="payload_too_large",
                        message="Streamed request body exceeds configured size limit.",
                        request_id=rid,
                    ),
                    request_id=rid,
                )
                return
            chunks.append(msg)
            more_body = bool(msg.get("more_body", False))

        idx = 0

        async def _replay_receive() -> Message:
            nonlocal idx
            if idx < len(chunks):
                item = chunks[idx]
                idx += 1
                return item
            return {"type": "http.request", "body": b"", "more_body": False}

        await self.app(scope, _replay_receive, send)


class SecurityHeadersMiddleware:
    def __init__(self, app: ASGIApp, settings: Settings) -> None:
        self.app = app
        self.settings = settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def _send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                hdrs = list(message.get("headers", ()))
                hdrs.extend(
                    [
                        (b"x-content-type-options", b"nosniff"),
                        (b"x-frame-options", b"DENY"),
                        (b"referrer-policy", b"no-referrer"),
                        (b"cross-origin-opener-policy", b"same-origin"),
                        (b"cross-origin-resource-policy", b"same-site"),
                        (b"content-security-policy", b"default-src 'none'; frame-ancestors 'none'"),
                    ]
                )
                if self.settings.production:
                    hdrs.append((b"strict-transport-security", b"max-age=31536000; includeSubDomains"))
                message = {**message, "headers": hdrs}
            await send(message)

        await self.app(scope, receive, _send_with_headers)


class InMemoryRateLimitMiddleware:
    """Trusted-proxy-aware sliding-window IP rate limiter emitting 429 + Retry-After + request_id."""

    def __init__(
        self,
        app: ASGIApp,
        requests_per_window: int,
        window_seconds: int,
        trusted_proxies: tuple[str, ...] = (),
    ) -> None:
        self.app = app
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.trusted_proxies = trusted_proxies
        self.client_windows: dict[str, deque[float]] = defaultdict(deque)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") in PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return

        headers = _headers_dict(scope)
        peer = scope.get("client")[0] if scope.get("client") else "unknown"
        client_host = extract_client_ip(peer, headers.get("x-forwarded-for"), self.trusted_proxies)
        now = time.monotonic()
        window = self.client_windows[client_host]
        while window and now - window[0] > self.window_seconds:
            window.popleft()

        if len(window) >= self.requests_per_window:
            retry_after = max(1, int(math.ceil(self.window_seconds - (now - window[0]))))
            rid = REQUEST_ID_CTX.get()
            if rid == "none":
                rid = str(uuid4())
            await _send_json(
                send,
                status.HTTP_429_TOO_MANY_REQUESTS,
                error_envelope(
                    status.HTTP_429_TOO_MANY_REQUESTS,
                    code="rate_limit_exceeded",
                    message="Rate limit exceeded.",
                    request_id=rid,
                ),
                request_id=rid,
                extra_headers=[(b"retry-after", str(retry_after).encode("ascii"))],
            )
            return

        window.append(now)
        await self.app(scope, receive, send)


class ConcurrencyCapMiddleware:
    """Bounds active compute requests cleanly via non-queueing concurrency cap returning 429 + Retry-After."""

    def __init__(self, app: ASGIApp, max_concurrent: int | None = None, retry_after_s: int = 15) -> None:
        self.app = app
        self.max_concurrent = max_concurrent
        self.retry_after_s = max(1, int(retry_after_s))
        self.active = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        method = str(scope.get("method", "GET")).upper()
        if method == "GET" or path in PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return

        effective_cap = max(
            1,
            int(self.max_concurrent if self.max_concurrent is not None else get_settings().max_concurrent_orchestrations),
        )
        if self.active >= effective_cap:
            rid = REQUEST_ID_CTX.get()
            if rid == "none":
                rid = str(uuid4())
            await _send_json(
                send,
                status.HTTP_429_TOO_MANY_REQUESTS,
                error_envelope(
                    status.HTTP_429_TOO_MANY_REQUESTS,
                    code="concurrency_limit_reached",
                    message=f"Server busy (max_concurrent={effective_cap}). Retry after {self.retry_after_s}s.",
                    request_id=rid,
                ),
                request_id=rid,
                extra_headers=[(b"retry-after", str(self.retry_after_s).encode("ascii"))],
            )
            return

        self.active += 1
        try:
            await self.app(scope, receive, send)
        finally:
            self.active -= 1


async def require_api_key(request: Request) -> None:
    settings = get_settings()
    if not settings.api_key_required:
        return

    if settings.production and not settings.disastermind_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
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
