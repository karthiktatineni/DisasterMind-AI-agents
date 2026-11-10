"""Real similarity search via Pinecone vector index. No local fallback."""

from __future__ import annotations

import time
from typing import Any

from backend.app.schemas import DisasterScenario
from backend.app.services.embedding_service import EmbeddingConfigError, EmbeddingService
from backend.app.services.pinecone_client import PineconeConfigError, PineconeVectorClient


class SimilarityService:
    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        pinecone_client: PineconeVectorClient | None = None,
    ):
        self._embedding_service = embedding_service
        self._pinecone_client = pinecone_client

    def _get_embedding_service(self) -> EmbeddingService:
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService()
        return self._embedding_service

    def _get_pinecone_client(self) -> PineconeVectorClient:
        if self._pinecone_client is None:
            self._pinecone_client = PineconeVectorClient()
        return self._pinecone_client

    async def search_scenario(
        self,
        scenario: DisasterScenario,
        top_k: int = 10,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            embedding_service = self._get_embedding_service()
            pinecone = self._get_pinecone_client()
        except (EmbeddingConfigError, PineconeConfigError) as exc:
            return {"status": "insufficient_data", "reason": str(exc)}

        query_text = EmbeddingService.build_scenario_query(scenario.model_dump())
        try:
            query_embedding = await embedding_service.embed_text(query_text, input_type="query")
            
            filter_conditions = {}
            if scenario.disaster_type:
                filter_conditions["disaster_type"] = {"$eq": scenario.disaster_type}
            if scenario.country:
                filter_conditions["country"] = {"$eq": scenario.country}
            
            filter_metadata = filter_conditions if filter_conditions else None

            rows = await pinecone.query(
                vector=query_embedding,
                top_k=top_k,
                filter_metadata=filter_metadata,
            )
        except Exception as exc:
            return {"status": "insufficient_data", "reason": str(exc)}

        if not rows:
            return {"status": "insufficient_data", "reason": "No vector matches returned."}

        matches = [
            {
                "id": row.get("id"),
                "event_name": row.get("event_name") or row.get("id"),
                "disaster_type": row.get("disaster_type"),
                "disaster_subtype": row.get("disaster_subtype"),
                "country": row.get("country"),
                "region": row.get("region"),
                "location": row.get("location"),
                "start_year": row.get("start_year"),
                "total_deaths": row.get("total_deaths"),
                "total_affected": row.get("total_affected"),
                "total_damage": row.get("total_damage"),
                "similarity": round(float(row.get("similarity", 0)), 4),
            }
            for row in rows
        ]

        return {
            "status": "ok",
            "matches": matches,
            "query_text": query_text,
            "execution_time_ms": round((time.perf_counter() - started) * 1000, 2),
            "source": "pinecone",
        }

    async def search_text(self, query: str, top_k: int = 10) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            embedding_service = self._get_embedding_service()
            pinecone = self._get_pinecone_client()
        except (EmbeddingConfigError, PineconeConfigError) as exc:
            return {"status": "insufficient_data", "reason": str(exc)}

        try:
            query_embedding = await embedding_service.embed_text(query, input_type="query")
            rows = await pinecone.query(vector=query_embedding, top_k=top_k)
        except Exception as exc:
            return {"status": "insufficient_data", "reason": str(exc)}

        if not rows:
            return {"status": "insufficient_data", "reason": "No vector matches returned."}

        return {
            "status": "ok",
            "matches": [
                {
                    "id": row.get("id"),
                    "event_name": row.get("event_name") or row.get("id"),
                    "disaster_type": row.get("disaster_type"),
                    "disaster_subtype": row.get("disaster_subtype"),
                    "country": row.get("country"),
                    "region": row.get("region"),
                    "location": row.get("location"),
                    "start_year": row.get("start_year"),
                    "total_deaths": row.get("total_deaths"),
                    "total_affected": row.get("total_affected"),
                    "total_damage": row.get("total_damage"),
                    "similarity": round(float(row.get("similarity", 0)), 4),
                }
                for row in rows
            ],
            "execution_time_ms": round((time.perf_counter() - started) * 1000, 2),
            "source": "pinecone",
        }
