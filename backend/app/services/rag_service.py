"""RAG layer: scenario → embedding → Pinecone vector search → context builder."""

from __future__ import annotations

from typing import Any

from backend.app.config import Settings, get_settings
from backend.app.schemas import DisasterScenario
from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.pinecone_client import PineconeVectorClient
from backend.app.services.retrieval_service import RetrievalService


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
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        self._embedding_service = embedding_service
        self._pinecone_client = pinecone_client

    async def retrieve(
        self,
        scenario: DisasterScenario,
        top_k: int = 10,
        *,
        year_min: int | None = None,
        year_max: int | None = None,
        similarity_threshold: float | None = None,
    ) -> dict[str, Any]:
        query_text = EmbeddingService.build_scenario_query(scenario.model_dump())
        retrieval = RetrievalService(
            settings=self.settings,
            embedding_service=self._embedding_service,
            pinecone_client=self._pinecone_client,
        )
        result = await retrieval.search(
            query_text,
            top_k=top_k,
            disaster_type=scenario.disaster_type or None,
            country=scenario.country or None,
            year_min=year_min,
            year_max=year_max,
            similarity_threshold=similarity_threshold,
        )
        if result.status != "ok":
            payload: dict[str, Any] = {
                "status": "insufficient_data",
                "reason": result.reason or "No disasters retrieved.",
                "threshold_used": result.threshold_used,
                "top_score": result.top_score,
            }
            if result.status == "no_data":
                payload["no_data"] = True
            return payload

        retrieved = [m.model_dump() for m in result.matches]
        sources = [
            {
                "id": item["id"],
                "label": item.get("display_label") or _format_event_label(item),
                "similarity": item["similarity"],
                "country": item.get("country"),
                "year": item.get("start_year"),
            }
            for item in retrieved
        ]

        return {
            "status": "ok",
            "retrieved_disasters": retrieved,
            "record_ids": result.record_ids,
            "sources": sources,
            "context": _build_context(retrieved),
            "query_text": query_text,
            "threshold_used": result.threshold_used,
            "top_score": result.top_score,
        }
