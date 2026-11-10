"""RAG layer: scenario → embedding → Pinecone vector search → context builder."""

from __future__ import annotations

from typing import Any

from backend.app.schemas import DisasterScenario
from backend.app.services.embedding_service import EmbeddingConfigError, EmbeddingService
from backend.app.services.pinecone_client import PineconeConfigError, PineconeVectorClient


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
        pinecone_client: PineconeVectorClient | None = None,
    ):
        self._embedding_service = embedding_service
        self._pinecone_client = pinecone_client

    async def retrieve(self, scenario: DisasterScenario, top_k: int = 10) -> dict[str, Any]:
        try:
            embedding_service = self._embedding_service or EmbeddingService()
            pinecone = self._pinecone_client or PineconeVectorClient()
        except (EmbeddingConfigError, PineconeConfigError) as exc:
            return {"status": "insufficient_data", "reason": str(exc)}

        query_text = EmbeddingService.build_scenario_query(scenario.model_dump())
        try:
            query_embedding = await embedding_service.embed_text(query_text)
            
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
