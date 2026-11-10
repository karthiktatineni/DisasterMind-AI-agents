"""Pinecone vector database client for disaster retrieval and indexing."""

from __future__ import annotations

import logging
from typing import Any

from pinecone import Pinecone

from backend.app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class PineconeConfigError(RuntimeError):
    pass


class PineconeRetrievalError(RuntimeError):
    pass


class PineconeVectorClient:
    def __init__(self, settings: Settings | None = None, pc: Pinecone | None = None):
        self.settings = settings or get_settings()
        if not self.settings.pinecone_api_key:
            raise PineconeConfigError("PINECONE_API_KEY is missing. Add it to .env.")
        if not self.settings.pinecone_index_name:
            raise PineconeConfigError("PINECONE_INDEX_NAME is missing. Add it to .env.")

        self._pc = pc or Pinecone(api_key=self.settings.pinecone_api_key)
        self._index = self._pc.Index(self.settings.pinecone_index_name)

    @property
    def index_name(self) -> str:
        return self.settings.pinecone_index_name

    @property
    def namespace(self) -> str:
        return self.settings.pinecone_namespace

    async def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
        namespace: str | None = None,
    ) -> list[dict[str, Any]]:
        target_namespace = namespace if namespace is not None else self.namespace
        try:
            query_kwargs: dict[str, Any] = {
                "vector": vector,
                "top_k": top_k,
                "include_metadata": True,
            }
            if target_namespace:
                query_kwargs["namespace"] = target_namespace
            if filter_metadata:
                query_kwargs["filter"] = filter_metadata

            response = self._index.query(**query_kwargs)
            results = []
            for match in response.get("matches", []):
                meta = match.get("metadata", {}) or {}
                row = {
                    "id": match.get("id"),
                    "similarity": round(float(match.get("score", 0.0)), 4),
                    **meta,
                }
                results.append(row)
            return results
        except Exception as exc:
            raise PineconeRetrievalError(f"Pinecone query failed: {exc}") from exc

    def describe_index_stats(self) -> dict[str, Any]:
        try:
            return self._index.describe_index_stats()
        except Exception as exc:
            raise PineconeRetrievalError(f"Failed to describe index stats: {exc}") from exc

    async def health_check(self) -> dict[str, Any]:
        try:
            stats = self.describe_index_stats()
            return {
                "status": "ok",
                "index_name": self.index_name,
                "total_vector_count": stats.get("total_vector_count", 0),
                "dimension": stats.get("dimension"),
                "namespaces": list(stats.get("namespaces", {}).keys()),
            }
        except Exception as exc:
            return {"status": "error", "reason": str(exc)}
