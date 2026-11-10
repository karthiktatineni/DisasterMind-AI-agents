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


