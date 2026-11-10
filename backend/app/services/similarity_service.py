"""Real similarity search via Pinecone vector index. No local fallback."""

from __future__ import annotations

import time
from typing import Any

from backend.app.config import Settings, get_settings
from backend.app.schemas import DisasterScenario
from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.pinecone_client import PineconeVectorClient
from backend.app.services.retrieval_service import RetrievalService


class SimilarityService:
    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        pinecone_client: PineconeVectorClient | None = None,
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        self._embedding_service = embedding_service
        self._pinecone_client = pinecone_client

    def _get_retrieval_service(self) -> RetrievalService:
        return RetrievalService(
            settings=self.settings,
            embedding_service=self._embedding_service,
            pinecone_client=self._pinecone_client,
        )

    async def _run_search(
        self,
        query_text: str,
        *,
        top_k: int,
        disaster_type: str | None = None,
        country: str | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
        similarity_threshold: float | None = None,
        include_query_text: bool = False,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        result = await self._get_retrieval_service().search(
            query_text,
            top_k=top_k,
            disaster_type=disaster_type,
            country=country,
            year_min=year_min,
            year_max=year_max,
            similarity_threshold=similarity_threshold,
        )
        if result.status != "ok":
            payload: dict[str, Any] = {
                "status": "insufficient_data",
                "reason": result.reason or "No vector matches returned.",
                "threshold_used": result.threshold_used,
                "top_score": result.top_score,
            }
            if result.status == "no_data":
                payload["no_data"] = True
            return payload

        matches = [m.model_dump() for m in result.matches]
        response: dict[str, Any] = {
            "status": "ok",
            "matches": matches,
            "record_ids": result.record_ids,
            "threshold_used": result.threshold_used,
            "top_score": result.top_score,
            "execution_time_ms": round((time.perf_counter() - started) * 1000, 2),
            "source": "pinecone",
        }
        if include_query_text:
            response["query_text"] = query_text
        return response

    async def search_scenario(
        self,
        scenario: DisasterScenario,
        top_k: int = 10,
        *,
        year_min: int | None = None,
        year_max: int | None = None,
        similarity_threshold: float | None = None,
    ) -> dict[str, Any]:
        query_text = EmbeddingService.build_scenario_query(scenario.model_dump())
        return await self._run_search(
            query_text,
            top_k=top_k,
            disaster_type=scenario.disaster_type or None,
            country=scenario.country or None,
            year_min=year_min,
            year_max=year_max,
            similarity_threshold=similarity_threshold,
            include_query_text=True,
        )

    async def search_text(
        self,
        query: str,
        top_k: int = 10,
        *,
        disaster_type: str | None = None,
        country: str | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
        similarity_threshold: float | None = None,
    ) -> dict[str, Any]:
        return await self._run_search(
            query,
            top_k=top_k,
            disaster_type=disaster_type,
            country=country,
            year_min=year_min,
            year_max=year_max,
            similarity_threshold=similarity_threshold,
            include_query_text=False,
        )
