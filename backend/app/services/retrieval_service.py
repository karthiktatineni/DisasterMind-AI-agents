"""Task 2P/3P Retrieval Service: async Pinecone retrieval with metadata filtering, calibrated threshold, and no_data path."""

from __future__ import annotations

from typing import Any

from backend.app.config import Settings, get_settings
from backend.app.schemas import DisasterScenario, RetrievalMatch, RetrievalResult
from backend.app.services.embedding_service import EmbeddingConfigError, EmbeddingService
from backend.app.services.pinecone_client import (
    PineconeConfigError,
    PineconeVectorClient,
    build_metadata_filter,
)


class RetrievalService:
    """Typed async retrieval service querying Pinecone with input_type='query' and calibrated threshold."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedding_service: EmbeddingService | None = None,
        pinecone_client: PineconeVectorClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._embedding_service = embedding_service
        self._pinecone_client = pinecone_client

    def _get_embedding_service(self) -> EmbeddingService:
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService(settings=self.settings)
        return self._embedding_service

    def _get_pinecone_client(self) -> PineconeVectorClient:
        if self._pinecone_client is None:
            self._pinecone_client = PineconeVectorClient(settings=self.settings)
        return self._pinecone_client

    async def search(
        self,
        query_text: str,
        *,
        top_k: int = 10,
        disaster_type: str | None = None,
        country: str | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
        similarity_threshold: float | None = None,
        namespace: str | None = None,
    ) -> RetrievalResult:
        threshold = self.settings.similarity_threshold if similarity_threshold is None else float(similarity_threshold)
        target_ns = namespace if namespace is not None else self.settings.pinecone_namespace

        try:
            embedding_svc = self._get_embedding_service()
            pinecone_svc = self._get_pinecone_client()
        except (EmbeddingConfigError, PineconeConfigError) as exc:
            return RetrievalResult(
                status="error",
                threshold_used=threshold,
                namespace=target_ns,
                reason=str(exc),
            )

        filter_metadata = build_metadata_filter(
            disaster_type=disaster_type,
            country=country,
            year_min=year_min,
            year_max=year_max,
        )
        filters_applied = {
            k: v
            for k, v in {
                "disaster_type": disaster_type,
                "country": country,
                "year_min": year_min,
                "year_max": year_max,
            }.items()
            if v is not None
        }

        try:
            query_vector = await embedding_svc.embed_text(query_text, input_type="query")
            rows = await pinecone_svc.query(
                vector=query_vector,
                top_k=top_k,
                filter_metadata=filter_metadata,
                namespace=target_ns,
            )
        except Exception as exc:
            return RetrievalResult(
                status="error",
                threshold_used=threshold,
                namespace=target_ns,
                filters_applied=filters_applied,
                reason=str(exc),
            )

        if not rows:
            return RetrievalResult(
                status="no_data",
                threshold_used=threshold,
                top_score=None,
                namespace=target_ns,
                filters_applied=filters_applied,
                reason="No vector matches returned.",
            )

        # Pinecone returns matches in descending cosine similarity order (use scores as-is)
        top_score = float(rows[0].get("similarity", 0.0))
        kept = [r for r in rows if float(r.get("similarity", 0.0)) >= threshold]

        if not kept:
            return RetrievalResult(
                status="no_data",
                threshold_used=threshold,
                top_score=top_score,
                namespace=target_ns,
                filters_applied=filters_applied,
                reason=f"Top similarity {top_score:.4f} below calibrated reject threshold {threshold:.4f}.",
            )

        matches: list[RetrievalMatch] = []
        record_ids: list[str] = []
        dataset_version: str | None = None
        for r in kept:
            rid = str(r.get("id"))
            record_ids.append(rid)
            if dataset_version is None and r.get("dataset_version"):
                dataset_version = str(r["dataset_version"])
            matches.append(
                RetrievalMatch(
                    id=rid,
                    event_name=str(r.get("event_name") or rid),
                    display_label=r.get("display_label"),
                    disaster_type=r.get("disaster_type"),
                    disaster_subtype=r.get("disaster_subtype"),
                    category_tag=r.get("category_tag"),
                    country=r.get("country"),
                    region=r.get("region"),
                    location=r.get("location"),
                    start_year=int(r["start_year"]) if r.get("start_year") is not None else None,
                    total_deaths=r.get("total_deaths"),
                    total_affected=r.get("total_affected"),
                    total_damage=r.get("total_damage"),
                    similarity=round(float(r.get("similarity", 0.0)), 4),
                    search_text=r.get("search_text"),
                )
            )

        return RetrievalResult(
            status="ok",
            record_ids=record_ids,
            matches=matches,
            threshold_used=threshold,
            top_score=top_score,
            namespace=target_ns,
            dataset_version=dataset_version,
            filters_applied=filters_applied,
        )

    async def search_scenario(
        self,
        scenario: DisasterScenario,
        *,
        top_k: int = 10,
        year_min: int | None = None,
        year_max: int | None = None,
    ) -> RetrievalResult:
        query_text = EmbeddingService.build_scenario_query(scenario.model_dump())
        return await self.search(
            query_text,
            top_k=top_k,
            disaster_type=scenario.disaster_type or None,
            country=scenario.country or None,
            year_min=year_min,
            year_max=year_max,
        )

