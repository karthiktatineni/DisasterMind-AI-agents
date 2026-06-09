"""Supabase / Postgres client for pgvector retrieval. No local fallback."""

from __future__ import annotations

from typing import Any

import httpx

from backend.app.config import Settings, get_settings


class SupabaseConfigError(RuntimeError):
    pass


class SupabaseRetrievalError(RuntimeError):
    pass


class SupabaseVectorClient:
    def __init__(self, settings: Settings | None = None, http_client: httpx.AsyncClient | None = None):
        self.settings = settings or get_settings()
        if not self.settings.supabase_url or not self.settings.supabase_service_role_key:
            raise SupabaseConfigError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required for vector retrieval."
            )
        self._base_url = self.settings.supabase_url.rstrip("/")
        self._headers = {
            "apikey": self.settings.supabase_service_role_key,
            "Authorization": f"Bearer {self.settings.supabase_service_role_key}",
            "Content-Type": "application/json",
        }
        self._client = http_client

    async def _request(self, method: str, path: str, json_body: dict | None = None) -> Any:
        url = f"{self._base_url}/rest/v1/{path}"
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=60.0)
        try:
            response = await client.request(method, url, headers=self._headers, json=json_body)
            if response.status_code >= 400:
                raise SupabaseRetrievalError(
                    f"Supabase request failed ({response.status_code}): {response.text[:500]}"
                )
            if not response.content:
                return None
            return response.json()
        finally:
            if owns_client:
                await client.aclose()

    async def rpc_match_disasters(
        self,
        query_embedding: list[float],
        match_count: int = 10,
        filter_disaster_type: str | None = None,
        filter_country: str | None = None,
    ) -> list[dict[str, Any]]:
        payload = {
            "query_embedding": query_embedding,
            "match_count": match_count,
            "filter_disaster_type": filter_disaster_type,
            "filter_country": filter_country,
        }
        result = await self._request("POST", "rpc/match_disasters", json_body=payload)
        if result is None:
            return []
        return list(result)

    async def count_rows(self) -> int:
        url = f"{self._base_url}/rest/v1/disaster_embeddings?select=id"
        headers = {**self._headers, "Prefer": "count=exact"}
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=30.0)
        try:
            response = await client.head(url, headers=headers)
            if response.status_code >= 400:
                raise SupabaseRetrievalError(f"Count failed: {response.status_code}")
            content_range = response.headers.get("content-range", "")
            if "/" in content_range:
                return int(content_range.split("/")[-1])
            return 0
        finally:
            if owns_client:
                await client.aclose()

    async def health_check(self) -> dict[str, Any]:
        row_count = await self.count_rows()
        return {
            "connected": True,
            "project_url": self._base_url,
            "disaster_embeddings_rows": row_count,
        }
