from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any

import httpx

from backend.app.config import Settings

logger = logging.getLogger(__name__)


class KeepAliveService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._tasks: list[asyncio.Task[None]] = []

    @property
    def backend_enabled(self) -> bool:
        return bool(self.settings.enable_backend_keepalive and self.settings.backend_keepalive_url)

    async def start(self) -> None:
        if self._tasks:
            return
        if self.backend_enabled:
            self._tasks.append(asyncio.create_task(self._backend_loop(), name="backend-keepalive"))

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()

    def status(self) -> dict[str, Any]:
        return {
            "backend_keepalive": {
                "enabled": self.backend_enabled,
                "url_configured": bool(self.settings.backend_keepalive_url),
                "interval_seconds": self.settings.backend_keepalive_interval_seconds,
            },
        }

    async def ping_backend(self) -> dict[str, Any]:
        url = self._backend_health_url()
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            return {"status": "ok", "url": url, "http_status": response.status_code}

    def _backend_health_url(self) -> str:
        base_url = self.settings.backend_keepalive_url.rstrip("/")
        if base_url.endswith("/health") or base_url.endswith("/keepalive"):
            return base_url
        return f"{base_url}/health"

    async def _backend_loop(self) -> None:
        await asyncio.sleep(5)
        while True:
            try:
                await self.ping_backend()
            except (httpx.HTTPError, RuntimeError) as exc:
                logger.warning("Backend keepalive ping failed: %s", exc)
            await asyncio.sleep(max(60, self.settings.backend_keepalive_interval_seconds))
