"""Real similarity search via Supabase pgvector. No local fallback."""

from __future__ import annotations

import time
from typing import Any

from backend.app.schemas import DisasterScenario
from backend.app.services.embedding_service import EmbeddingConfigError, EmbeddingService
from backend.app.services.supabase_client import SupabaseConfigError, SupabaseVectorClient


class SimilarityService:
    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        supabase_client: SupabaseVectorClient | None = None,
    ):
        self._embedding_service = embedding_service
        self._supabase_client = supabase_client

    def _get_embedding_service(self) -> EmbeddingService:
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService()
        return self._embedding_service

    def _get_supabase_client(self) -> SupabaseVectorClient:
        if self._supabase_client is None:
            self._supabase_client = SupabaseVectorClient()
        return self._supabase_client

    async def search_scenario(
        self,
        scenario: DisasterScenario,
        top_k: int = 10,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            embedding_service = self._get_embedding_service()
            supabase = self._get_supabase_client()
        except (EmbeddingConfigError, SupabaseConfigError) as exc:
            return {"status": "insufficient_data", "reason": str(exc)}

        query_text = EmbeddingService.build_scenario_query(scenario.model_dump())
        try:
            query_embedding = await embedding_service.embed_text(query_text)
            filter_type = scenario.disaster_type if scenario.disaster_type else None
            filter_country = scenario.country if scenario.country else None
            rows = await supabase.rpc_match_disasters(
                query_embedding=query_embedding,
                match_count=top_k,
                filter_disaster_type=filter_type,
                filter_country=filter_country,
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
            "source": "supabase_pgvector",
        }

    async def search_text(self, query: str, top_k: int = 10) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            embedding_service = self._get_embedding_service()
            supabase = self._get_supabase_client()
        except (EmbeddingConfigError, SupabaseConfigError) as exc:
            return {"status": "insufficient_data", "reason": str(exc)}

        try:
            query_embedding = await embedding_service.embed_text(query)
            rows = await supabase.rpc_match_disasters(query_embedding=query_embedding, match_count=top_k)
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
            "source": "supabase_pgvector",
        }
