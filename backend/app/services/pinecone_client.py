"""Async Pinecone vector database client for disaster retrieval and indexing."""

from __future__ import annotations

import inspect
import logging
from typing import Any

from pinecone import AsyncPinecone, Pinecone

from backend.app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class PineconeConfigError(RuntimeError):
    pass


class PineconeRetrievalError(RuntimeError):
    pass


def build_metadata_filter(
    *,
    disaster_type: str | None = None,
    country: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
) -> dict[str, Any] | None:
    """Build a Pinecone metadata filter dictionary from optional hazard, country, and year bounds."""
    clauses: list[dict[str, Any]] = []
    if disaster_type and disaster_type.strip():
        clauses.append({"disaster_type": {"$eq": disaster_type.strip()}})
    if country and country.strip():
        clauses.append({"country": {"$eq": country.strip()}})
    if year_min is not None:
        clauses.append({"start_year": {"$gte": int(year_min)}})
    if year_max is not None:
        clauses.append({"start_year": {"$lte": int(year_max)}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


class PineconeVectorClient:
    """Non-blocking AsyncPinecone client (with sync-mock injection support for unit tests)."""

    def __init__(
        self,
        settings: Settings | None = None,
        pc: Pinecone | AsyncPinecone | Any | None = None,
        *,
        timeout: float = 10.0,
        host: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        if not self.settings.pinecone_api_key:
            raise PineconeConfigError("PINECONE_API_KEY is missing. Add it to .env.")
        if not self.settings.pinecone_index_name:
            raise PineconeConfigError("PINECONE_INDEX_NAME is missing. Add it to .env.")

        self.timeout = timeout
        self._host = host
        self._injected_pc = pc
        self._async_pc: AsyncPinecone | None = None
        self._async_index: Any = None
        self._sync_index = pc.Index(self.settings.pinecone_index_name) if (pc is not None and hasattr(pc, "Index")) else None

    @property
    def index_name(self) -> str:
        return self.settings.pinecone_index_name

    @property
    def namespace(self) -> str:
        return self.settings.pinecone_namespace

    async def _get_index(self) -> Any:
        if self._sync_index is not None:
            return self._sync_index
        if self._async_index is not None:
            return self._async_index

        if self._injected_pc is not None and hasattr(self._injected_pc, "IndexAsyncio"):
            if not self._host:
                desc = await self._injected_pc.describe_index(self.index_name)
                self._host = desc.host
            self._async_index = self._injected_pc.IndexAsyncio(host=self._host)
            return self._async_index

        self._async_pc = AsyncPinecone(
            api_key=self.settings.pinecone_api_key,
            timeout=self.timeout,
        )
        if not self._host:
            desc = await self._async_pc.describe_index(self.index_name)
            self._host = desc.host
        self._async_index = self._async_pc.IndexAsyncio(host=self._host)
        return self._async_index

    async def aclose(self) -> None:
        if self._async_index is not None and hasattr(self._async_index, "close"):
            await self._async_index.close()
            self._async_index = None
        if self._async_pc is not None and hasattr(self._async_pc, "close"):
            await self._async_pc.close()
            self._async_pc = None

    async def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
        namespace: str | None = None,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        target_namespace = namespace if namespace is not None else self.namespace
        try:
            index = await self._get_index()
            query_kwargs: dict[str, Any] = {
                "vector": vector,
                "top_k": top_k,
                "include_metadata": True,
            }
            if target_namespace:
                query_kwargs["namespace"] = target_namespace
            if filter_metadata:
                query_kwargs["filter"] = filter_metadata
            if timeout is not None and self._sync_index is None:
                query_kwargs["timeout"] = timeout

            raw_response = index.query(**query_kwargs)
            response = await raw_response if inspect.isawaitable(raw_response) else raw_response

            matches = response.get("matches", []) if isinstance(response, dict) else getattr(response, "matches", [])
            results: list[dict[str, Any]] = []
            for match in matches:
                if isinstance(match, dict):
                    mid = match.get("id")
                    mscore = match.get("score", 0.0)
                    meta = match.get("metadata") or {}
                else:
                    mid = getattr(match, "id", None)
                    mscore = getattr(match, "score", 0.0)
                    meta = getattr(match, "metadata", None) or {}
                # Invariant D5: never return a retrieval result without metadata
                if not meta:
                    logger.warning("Skipping vector match '%s' due to empty metadata (D5).", mid)
                    continue
                results.append(
                    {
                        "id": mid,
                        "similarity": round(float(mscore), 4),
                        **meta,
                    }
                )
            return results
        except Exception as exc:
            raise PineconeRetrievalError(f"Pinecone query failed: {exc}") from exc

    async def describe_index_stats_async(self) -> dict[str, Any]:
        try:
            index = await self._get_index()
            raw_stats = index.describe_index_stats()
            stats = await raw_stats if inspect.isawaitable(raw_stats) else raw_stats
            if hasattr(stats, "to_dict"):
                return stats.to_dict()
            return dict(stats)
        except Exception as exc:
            raise PineconeRetrievalError(f"Failed to describe index stats: {exc}") from exc

    def describe_index_stats(self) -> dict[str, Any]:
        if self._sync_index is not None:
            try:
                return self._sync_index.describe_index_stats()
            except Exception as exc:
                raise PineconeRetrievalError(f"Failed to describe index stats: {exc}") from exc
        try:
            pc = Pinecone(api_key=self.settings.pinecone_api_key, timeout=self.timeout)
            idx = pc.Index(self.index_name)
            stats = idx.describe_index_stats()
            return stats.to_dict() if hasattr(stats, "to_dict") else dict(stats)
        except Exception as exc:
            raise PineconeRetrievalError(f"Failed to describe index stats: {exc}") from exc

    async def health_check(self) -> dict[str, Any]:
        try:
            stats = await self.describe_index_stats_async()
            return {
                "status": "ok",
                "index_name": self.index_name,
                "total_vector_count": stats.get("total_vector_count", 0),
                "dimension": stats.get("dimension"),
                "namespaces": list(stats.get("namespaces", {}).keys()),
            }
        except Exception as exc:
            return {"status": "error", "reason": str(exc)}
