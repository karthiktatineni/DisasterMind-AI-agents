"""RAG layer: scenario → embedding → vector search → context builder."""

from __future__ import annotations

from typing import Any

from backend.app.schemas import DisasterScenario
from backend.app.services.embedding_service import EmbeddingConfigError, EmbeddingService
from backend.app.services.supabase_client import SupabaseConfigError, SupabaseVectorClient


def _format_event_label(row: dict[str, Any]) -> str:
    name = row.get("event_name") or row.get("id") or "Unknown event"
    year = row.get("start_year")
    return f"{name} ({year})" if year else str(name)


def _build_context(rows: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for index, row in enumerate(rows, start=1):
        similarity = round(float(row.get("similarity", 0)), 4)
        blocks.append(
            f"[{index}] {_format_event_label(row)} | "
            f"Type: {row.get('disaster_type')} | Country: {row.get('country')} | "
            f"Deaths: {row.get('total_deaths')} | Affected: {row.get('total_affected')} | "
            f"Damage (USD thousands): {row.get('total_damage')} | Similarity: {similarity}\n"
            f"{row.get('search_text', '')}"
        )
    return "\n\n".join(blocks)


class RAGService:
    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        supabase_client: SupabaseVectorClient | None = None,
    ):
        self._embedding_service = embedding_service
        self._supabase_client = supabase_client

    async def retrieve(self, scenario: DisasterScenario, top_k: int = 10) -> dict[str, Any]:
        try:
            embedding_service = self._embedding_service or EmbeddingService()
            supabase = self._supabase_client or SupabaseVectorClient()
        except (EmbeddingConfigError, SupabaseConfigError) as exc:
            return {"status": "insufficient_data", "reason": str(exc)}

        query_text = EmbeddingService.build_scenario_query(scenario.model_dump())
        try:
            query_embedding = await embedding_service.embed_text(query_text)
            rows = await supabase.rpc_match_disasters(
                query_embedding=query_embedding,
                match_count=top_k,
                filter_disaster_type=scenario.disaster_type or None,
                filter_country=scenario.country or None,
            )
        except Exception as exc:
            return {"status": "insufficient_data", "reason": str(exc)}

        if not rows:
            return {"status": "insufficient_data", "reason": "No disasters retrieved."}

        retrieved = [
            {
                "id": row.get("id"),
                "event_name": row.get("event_name") or row.get("id"),
                "disaster_type": row.get("disaster_type"),
                "country": row.get("country"),
                "start_year": row.get("start_year"),
                "total_deaths": row.get("total_deaths"),
                "total_affected": row.get("total_affected"),
                "total_damage": row.get("total_damage"),
                "similarity": round(float(row.get("similarity", 0)), 4),
                "search_text": row.get("search_text"),
            }
            for row in rows
        ]

        sources = [
            {
                "id": item["id"],
                "label": _format_event_label(item),
                "similarity": item["similarity"],
                "country": item.get("country"),
                "year": item.get("start_year"),
            }
            for item in retrieved
        ]

        return {
            "status": "ok",
            "retrieved_disasters": retrieved,
            "sources": sources,
            "context": _build_context(rows),
            "query_text": query_text,
        }
