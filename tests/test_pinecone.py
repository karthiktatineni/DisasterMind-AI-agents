"""Unit tests for Pinecone vector client and configuration validation."""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from backend.app.config import Settings
from backend.app.services.pinecone_client import (
    PineconeConfigError,
    PineconeVectorClient,
)


def test_pinecone_config_error_when_missing_key():
    settings = Settings(
        nvidia_api_key="test-key",
        pinecone_api_key="",
        pinecone_index_name="disastermind",
    )
    with pytest.raises(PineconeConfigError, match="PINECONE_API_KEY is missing"):
        PineconeVectorClient(settings=settings)


def test_pinecone_config_error_when_missing_index_name():
    settings = Settings(
        nvidia_api_key="test-key",
        pinecone_api_key="pcsk_test",
        pinecone_index_name="",
    )
    with pytest.raises(PineconeConfigError, match="PINECONE_INDEX_NAME is missing"):
        PineconeVectorClient(settings=settings)


@pytest.mark.asyncio
async def test_pinecone_query_maps_matches_with_metadata():
    settings = Settings(
        nvidia_api_key="test-key",
        pinecone_api_key="pcsk_test",
        pinecone_index_name="disastermind",
        pinecone_namespace="v1-test",
    )
    mock_pc = MagicMock()
    mock_index = MagicMock()
    mock_pc.Index.return_value = mock_index

    mock_index.query.return_value = {
        "matches": [
            {
                "id": "2020-001-PHL",
                "score": 0.9421,
                "metadata": {
                    "event_name": "Typhoon Goni",
                    "disaster_type": "Storm",
                    "country": "Philippines",
                    "start_year": 2020,
                    "total_deaths": 25,
                },
            }
        ]
    }

    client = PineconeVectorClient(settings=settings, pc=mock_pc)
    results = await client.query(vector=[0.1] * 2048, top_k=5)

    assert len(results) == 1
    assert results[0]["id"] == "2020-001-PHL"
    assert results[0]["similarity"] == 0.9421
    assert results[0]["event_name"] == "Typhoon Goni"
    assert results[0]["country"] == "Philippines"

    # Verify query arguments
    mock_index.query.assert_called_once_with(
        vector=[0.1] * 2048,
        top_k=5,
        include_metadata=True,
        namespace="v1-test",
    )


@pytest.mark.asyncio
async def test_pinecone_health_check_reports_stats():
    settings = Settings(
        nvidia_api_key="test-key",
        embedding_provider="nvidia",
        embedding_model="nvidia/nemotron-3-embed-1b",
        embedding_dimension=2048,
        pinecone_api_key="pcsk_test",
        pinecone_index_name="disastermind",
    )
    mock_pc = MagicMock()
    mock_index = MagicMock()
    mock_pc.Index.return_value = mock_index

    mock_index.describe_index_stats.return_value = {
        "dimension": 2048,
        "total_vector_count": 16000,
        "namespaces": {"v1-test": {"vector_count": 16000}},
    }

    client = PineconeVectorClient(settings=settings, pc=mock_pc)
    health = await client.health_check()

    assert health["status"] == "ok"
    assert health["index_name"] == "disastermind"
    assert health["dimension"] == 2048
    assert health["total_vector_count"] == 16000
    assert "v1-test" in health["namespaces"]


@pytest.mark.asyncio
async def test_pinecone_scores_direct_descending_no_inversion():
    """Verify similarity scores come straight from Pinecone scores in descending order (no 1 - score inversion)."""
    settings = Settings(
        nvidia_api_key="test-key",
        embedding_provider="nvidia",
        embedding_model="nvidia/nemotron-3-embed-1b",
        embedding_dimension=2048,
        pinecone_api_key="pcsk_test",
        pinecone_index_name="disastermind",
    )
    mock_pc = MagicMock()
    mock_index = MagicMock()
    mock_pc.Index.return_value = mock_index

    # Pinecone returns matches in descending score order
    mock_index.query.return_value = {
        "matches": [
            {"id": "A", "score": 0.95, "metadata": {"event_name": "Event A"}},
            {"id": "B", "score": 0.85, "metadata": {"event_name": "Event B"}},
            {"id": "C", "score": 0.70, "metadata": {"event_name": "Event C"}},
        ]
    }

    client = PineconeVectorClient(settings=settings, pc=mock_pc)
    results = await client.query(vector=[0.0] * 2048, top_k=3)

    assert len(results) == 3
    # Assert exact descending order
    scores = [r["similarity"] for r in results]
    assert scores == [0.95, 0.85, 0.70]
    # Assert no '1 - x' transformation was applied
    assert scores[0] == 0.95 and scores[0] != round(1.0 - 0.95, 4)
    assert scores[1] == 0.85 and scores[1] != round(1.0 - 0.85, 4)
    assert scores[2] == 0.70 and scores[2] != round(1.0 - 0.70, 4)


@pytest.mark.asyncio
async def test_similarity_and_rag_preserve_pinecone_scores_and_query_input_type():
    """Verify SimilarityService and RAGService forward raw Pinecone scores descending and pass input_type='query'."""
    from types import SimpleNamespace
    from backend.app.schemas import DisasterScenario
    from backend.app.services.embedding_service import EmbeddingService
    from backend.app.services.rag_service import RAGService
    from backend.app.services.similarity_service import SimilarityService

    recorded_embed_calls = []

    class FakeEmbeddingsAPI:
        async def create(self, **kwargs):
            recorded_embed_calls.append(kwargs)
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.01] * 2048)])

    fake_openai = SimpleNamespace(embeddings=FakeEmbeddingsAPI())
    settings = Settings(
        nvidia_api_key="test-key",
        embedding_provider="nvidia",
        embedding_model="nvidia/nemotron-3-embed-1b",
        embedding_dimension=2048,
        pinecone_api_key="pcsk_test",
        pinecone_index_name="disastermind",
    )
    emb_service = EmbeddingService(settings=settings, client=fake_openai)

    mock_pc = MagicMock()
    mock_index = MagicMock()
    mock_pc.Index.return_value = mock_index
    mock_index.query.return_value = {
        "matches": [
            {"id": "2019-1", "score": 0.9123, "metadata": {"event_name": "Storm Alpha", "start_year": 2019}},
            {"id": "2018-2", "score": 0.7654, "metadata": {"event_name": "Storm Beta", "start_year": 2018}},
            {"id": "2015-3", "score": 0.6011, "metadata": {"event_name": "Storm Gamma", "start_year": 2015}},
        ]
    }
    pc_client = PineconeVectorClient(settings=settings, pc=mock_pc)

    sim_service = SimilarityService(embedding_service=emb_service, pinecone_client=pc_client)
    rag_service = RAGService(embedding_service=emb_service, pinecone_client=pc_client)
    scenario = DisasterScenario(disaster_type="Storm", region="Odisha", country="India", severity="High")

    sim_res = await sim_service.search_scenario(scenario, top_k=3)
    sim_text_res = await sim_service.search_text("cyclone in India", top_k=3)
    rag_res = await rag_service.retrieve(scenario, top_k=3)

    assert sim_res["status"] == "ok"
    assert [m["similarity"] for m in sim_res["matches"]] == [0.9123, 0.7654, 0.6011]
    assert [m["similarity"] for m in sim_text_res["matches"]] == [0.9123, 0.7654, 0.6011]
    assert [d["similarity"] for d in rag_res["retrieved_disasters"]] == [0.9123, 0.7654, 0.6011]
    assert all(call["extra_body"]["input_type"] == "query" for call in recorded_embed_calls)


@pytest.mark.asyncio
async def test_embed_text_requires_input_type_and_retries_5xx(monkeypatch):
    """Verify input_type is required keyword-only and 5xx errors trigger exponential backoff retry."""
    from types import SimpleNamespace
    import httpx
    from openai import APIStatusError
    from backend.app.services.embedding_service import EmbeddingService

    sleep_delays = []

    async def fake_sleep(delay):
        sleep_delays.append(delay)

    monkeypatch.setattr("backend.app.services.embedding_service.asyncio.sleep", fake_sleep)

    attempts = {"count": 0}
    recorded_calls = []

    class FlakyEmbeddingsAPI:
        async def create(self, **kwargs):
            recorded_calls.append(kwargs)
            attempts["count"] += 1
            if attempts["count"] <= 2:
                req = httpx.Request("POST", "https://integrate.api.nvidia.com/v1/embeddings")
                resp = httpx.Response(500, request=req)
                raise APIStatusError("service inference failed", response=resp, body={"message": "server_error"})
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.02] * 2048)])

    fake_openai = SimpleNamespace(embeddings=FlakyEmbeddingsAPI())
    settings = Settings(
        nvidia_api_key="test-key",
        embedding_provider="nvidia",
        embedding_model="nvidia/nemotron-3-embed-1b",
        embedding_dimension=2048,
    )
    emb = EmbeddingService(settings=settings, client=fake_openai)

    with pytest.raises(TypeError):
        await emb.embed_text("missing input_type")  # type: ignore[call-arg]

    with pytest.raises(ValueError, match="input_type must be"):
        await emb.embed_text("invalid input_type", input_type="invalid")

    vecs = await emb.embed_texts(["passage text"], input_type="passage", max_retries=4, base_delay=0.5)
    assert len(vecs) == 1
    assert len(vecs[0]) == 2048
    assert len(sleep_delays) == 2
    assert recorded_calls[-1]["extra_body"]["input_type"] == "passage"


@pytest.mark.asyncio
async def test_embedding_service_pinecone_inference_provider(monkeypatch):
    """Verify Pinecone-hosted models route to Pinecone Inference with input_type and retry on transient errors."""
    from types import SimpleNamespace
    from backend.app.services.embedding_service import EmbeddingService

    sleep_delays = []

    async def fake_sleep(delay):
        sleep_delays.append(delay)

    monkeypatch.setattr("backend.app.services.embedding_service.asyncio.sleep", fake_sleep)

    recorded = []
    attempts = {"n": 0}

    class FakeInference:
        async def embed(self, *, model, inputs, parameters=None):
            recorded.append({"model": model, "inputs": inputs, "parameters": parameters})
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("transient 503")
            return SimpleNamespace(data=[SimpleNamespace(values=[0.25] * 1024) for _ in inputs])

    fake_pc = SimpleNamespace(inference=FakeInference())
    settings = Settings(
        nvidia_api_key="test-key",
        embedding_provider="pinecone",
        embedding_model="llama-text-embed-v2",
        embedding_dimension=1024,
        pinecone_api_key="pcsk_test",
    )
    emb = EmbeddingService(settings=settings, pinecone_client=fake_pc)
    assert emb.provider == "pinecone-inference"

    qvec = await emb.embed_text("cyclone Odisha", input_type="query")
    assert len(qvec) == 1024
    assert len(sleep_delays) == 1
    assert recorded[-1]["model"] == "llama-text-embed-v2"
    assert recorded[-1]["parameters"]["input_type"] == "query"

    pvecs = await emb.embed_texts(["record one"], input_type="passage")
    assert len(pvecs) == 1
    assert recorded[-1]["parameters"]["input_type"] == "passage"


@pytest.mark.asyncio
async def test_retrieval_service_filters_threshold_no_data_and_record_ids():
    """Verify RetrievalService metadata filter construction, calibrated threshold reject (no_data), and record_ids."""
    from types import SimpleNamespace
    from backend.app.services.embedding_service import EmbeddingService
    from backend.app.services.rag_service import RAGService
    from backend.app.services.retrieval_service import RetrievalService
    from backend.app.services.similarity_service import SimilarityService
    from backend.app.schemas import DisasterScenario

    class FakeInference:
        async def embed(self, *, model, inputs, parameters=None):
            return SimpleNamespace(data=[SimpleNamespace(values=[0.1] * 1024) for _ in inputs])

    settings = Settings(
        nvidia_api_key="test-key",
        embedding_provider="pinecone",
        embedding_model="llama-text-embed-v2",
        embedding_dimension=1024,
        pinecone_api_key="pcsk_test",
        pinecone_index_name="disastermind",
        pinecone_namespace="v2-test",
        similarity_threshold=0.35,
    )
    emb = EmbeddingService(settings=settings, pinecone_client=SimpleNamespace(inference=FakeInference()))

    mock_pc = MagicMock()
    mock_index = MagicMock()
    mock_pc.Index.return_value = mock_index
    mock_index.query.return_value = {
        "matches": [
            {"id": "2019-1", "score": 0.62, "metadata": {"event_name": "Cyclone Fani", "disaster_type": "Storm", "country": "India", "start_year": 2019}},
            {"id": "1999-2", "score": 0.28, "metadata": {"event_name": "Below Threshold", "disaster_type": "Storm", "country": "India", "start_year": 1999}},
        ]
    }
    pc_client = PineconeVectorClient(settings=settings, pc=mock_pc)
    retrieval = RetrievalService(settings=settings, embedding_service=emb, pinecone_client=pc_client)

    res = await retrieval.search(
        "Storm in India",
        top_k=5,
        disaster_type="Storm",
        country="India",
        year_min=2000,
        year_max=2024,
    )
    assert res.status == "ok"
    assert res.record_ids == ["2019-1"]
    assert res.top_score == pytest.approx(0.62)
    assert res.threshold_used == pytest.approx(0.35)
    assert mock_index.query.call_args.kwargs["filter"] == {
        "$and": [
            {"disaster_type": {"$eq": "Storm"}},
            {"country": {"$eq": "India"}},
            {"start_year": {"$gte": 2000}},
            {"start_year": {"$lte": 2024}},
        ]
    }

    # Out-of-domain / low-similarity score triggers explicit no_data
    mock_index.query.return_value = {
        "matches": [
            {"id": "2010-9", "score": 0.21, "metadata": {"event_name": "Weak Match", "start_year": 2010}},
        ]
    }
    ood = await retrieval.search("chocolate cake recipe", top_k=5)
    assert ood.status == "no_data"
    assert ood.record_ids == []

    sim = SimilarityService(embedding_service=emb, pinecone_client=pc_client, settings=settings)
    rag = RAGService(embedding_service=emb, pinecone_client=pc_client, settings=settings)
    scenario = DisasterScenario(disaster_type="Storm", region="Odisha", country="India", severity="High")
    sim_ood = await sim.search_scenario(scenario, top_k=5)
    rag_ood = await rag.retrieve(scenario, top_k=5)
    assert sim_ood["status"] == "insufficient_data" and sim_ood["no_data"] is True
    assert rag_ood["status"] == "insufficient_data" and rag_ood["no_data"] is True



