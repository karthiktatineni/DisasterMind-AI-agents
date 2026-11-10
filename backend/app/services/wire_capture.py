"""Opt-in, secret-redacting HTTP wire capture for independent verification."""
from __future__ import annotations

import contextvars
import gzip
import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from backend.app.security import REQUEST_ID_CTX

ACTIVE_AGENT_CTX: contextvars.ContextVar[str | None] = contextvars.ContextVar("wire_agent", default=None)
_ORIGINAL_ASYNC_TRANSPORT = httpx.AsyncHTTPTransport
_WRITE_LOCK = threading.Lock()
_SECRET_RE = re.compile(r"(?i)(nvapi-[A-Za-z0-9_-]{10,}|pcsk_[A-Za-z0-9_-]{10,}|sb_secret_[A-Za-z0-9_-]{10,})")
_SECRET_HEADERS = {"authorization", "api-key", "x-api-key", "cookie", "set-cookie", "proxy-authorization"}
_SECRET_BODY_KEYS = {"api_key", "apikey", "access_token", "refresh_token", "authorization", "secret", "password"}


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): ("[REDACTED_SECRET]" if str(k).lower() in _SECRET_BODY_KEYS else _redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return _SECRET_RE.sub("[REDACTED_SECRET]", value)
    return value


def _safe_body(raw: bytes, headers: httpx.Headers) -> Any:
    content = raw
    encoding = headers.get("content-encoding", "").lower()
    try:
        if encoding == "gzip":
            content = gzip.decompress(raw)
        elif encoding == "deflate":
            import zlib
            content = zlib.decompress(raw)
        text = content.decode("utf-8")
    except Exception:
        return {"bytes": len(raw), "content_encoding": encoding or None}
    try:
        return _redact(json.loads(text))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _SECRET_RE.sub("[REDACTED_SECRET]", text[:200_000])


def _safe_headers(headers: httpx.Headers) -> dict[str, str]:
    return {k.lower(): ("[REDACTED_SECRET]" if k.lower() in _SECRET_HEADERS else v) for k, v in headers.items()}


def _write(record: dict[str, Any]) -> None:
    target = os.getenv("DISASTERMIND_WIRE_CAPTURE_PATH", "").strip()
    if not target:
        return
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK, path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")


class _TeeRequestStream(httpx.AsyncByteStream):
    def __init__(self, inner: httpx.AsyncByteStream) -> None:
        self.inner = inner
        self.parts: list[bytes] = []

    async def __aiter__(self):
        async for part in self.inner:
            self.parts.append(bytes(part))
            yield part

    async def aclose(self) -> None:
        await self.inner.aclose()


class _CaptureResponseStream(httpx.AsyncByteStream):
    def __init__(self, inner: httpx.AsyncByteStream, finish) -> None:
        self.inner = inner
        self.finish = finish
        self.parts: list[bytes] = []
        self.complete = False

    async def __aiter__(self):
        try:
            async for part in self.inner:
                self.parts.append(bytes(part))
                yield part
            self.complete = True
        finally:
            self.finish(b"".join(self.parts), self.complete)

    async def aclose(self) -> None:
        try:
            await self.inner.aclose()
        finally:
            if not self.complete:
                self.finish(b"".join(self.parts), False)


class WireSpyAsyncHTTPTransport(_ORIGINAL_ASYNC_TRANSPORT):
    """Capture real httpx traffic without consuming or changing response bytes."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        wire_id = str(uuid4())
        started = time.perf_counter()
        request_started = datetime.now(timezone.utc).isoformat()
        tee = _TeeRequestStream(request.stream)
        request.stream = tee
        request_id = REQUEST_ID_CTX.get()
        if request_id == "none":
            request_id = None
        agent = ACTIVE_AGENT_CTX.get()
        try:
            response = await super().handle_async_request(request)
        except Exception as exc:
            _write({
                "kind": "http_error", "wire_id": wire_id, "timestamp": request_started,
                "request_id": request_id, "agent": agent, "method": request.method,
                "host": request.url.host, "path": request.url.path,
                "request_headers": _safe_headers(request.headers),
                "request_body": _safe_body(b"".join(tee.parts), request.headers),
                "error_type": type(exc).__name__, "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            })
            raise

        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        request_record = {
            "kind": "http_request", "wire_id": wire_id, "timestamp": request_started,
            "request_id": request_id, "agent": agent, "method": request.method,
            "scheme": request.url.scheme, "host": request.url.host, "path": request.url.path,
            "request_headers": _safe_headers(request.headers),
            "request_body": _safe_body(b"".join(tee.parts), request.headers),
        }
        _write(request_record)

        def finish(body: bytes, complete: bool) -> None:
            _write({
                "kind": "http_response", "wire_id": wire_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_id": request_id, "agent": agent,
                "host": request.url.host, "path": request.url.path,
                "status": response.status_code,
                "response_headers": _safe_headers(response.headers),
                "response_body": _safe_body(body, response.headers),
                "elapsed_ms": elapsed_ms, "body_complete": complete,
            })

        response.stream = _CaptureResponseStream(response.stream, finish)
        return response


def enable_wire_capture() -> None:
    """Patch httpx's transport factory only when an explicit capture path is set."""
    if os.getenv("DISASTERMIND_WIRE_CAPTURE_PATH", "").strip():
        httpx.AsyncHTTPTransport = WireSpyAsyncHTTPTransport  # type: ignore[assignment]
        # AsyncClient resolves this module-global, not the re-export above.
        # Patch both so clients created by httpx and SDK wrappers are observed.
        httpx._client.AsyncHTTPTransport = WireSpyAsyncHTTPTransport  # type: ignore[attr-defined,assignment]
