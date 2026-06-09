from __future__ import annotations

import pytest

from backend.app.schemas import DisasterScenario, ScenarioIntakeRequest
from backend.app.services.orchestration import AGENT_SEQUENCE, DisasterMindOrchestrator


MOCK_RETRIEVAL = {
    "status": "ok",
    "retrieved_disasters": [
        {
            "id": "2019-0001-IND",
            "event_name": "Cyclone Fani",
            "disaster_type": "Storm",
            "country": "India",
            "start_year": 2019,
            "total_deaths": 89.0,
            "total_affected": 13000000.0,
            "total_damage": 8000000.0,
            "similarity": 0.91,
            "search_text": "Event: Cyclone Fani",
        },
        {
            "id": "2014-0002-IND",
            "event_name": "Cyclone Hudhud",
            "disaster_type": "Storm",
            "country": "India",
            "start_year": 2014,
            "total_deaths": 124.0,
            "total_affected": 9200000.0,
            "total_damage": 7000000.0,
            "similarity": 0.88,
            "search_text": "Event: Cyclone Hudhud",
        },
    ],
    "sources": [],
    "context": "mock context",
}

MOCK_SIMILARITY = {
    "status": "ok",
    "matches": [
        {"id": "2019-0001-IND", "event_name": "Cyclone Fani", "country": "India", "start_year": 2019, "similarity": 0.91},
        {"id": "2014-0002-IND", "event_name": "Cyclone Hudhud", "country": "India", "start_year": 2014, "similarity": 0.88},
    ],
    "execution_time_ms": 12.0,
    "source": "supabase_pgvector",
}


class MockSimilarityService:
    async def search_scenario(self, scenario, top_k=10):
        return MOCK_SIMILARITY

    async def search_text(self, query, top_k=10):
        return MOCK_SIMILARITY


class MockRAGService:
    async def retrieve(self, scenario, top_k=10):
        return MOCK_RETRIEVAL


@pytest.fixture
def orchestrator():
    return DisasterMindOrchestrator(
        similarity_service=MockSimilarityService(),
        rag_service=MockRAGService(),
    )


@pytest.mark.asyncio
async def test_orchestration_executes_all_agents_in_required_order(orchestrator, tmp_path, monkeypatch):
    from pathlib import Path
    import joblib
    import numpy as np
    from xgboost import XGBRegressor

    models_dir = Path(__file__).resolve().parents[1] / "models" / "xgboost"
    models_dir.mkdir(parents=True, exist_ok=True)
    X = np.array([[100, 200, 3000, 10, 80, 0.5, 100000, 2000, 50000]])
    y = np.array([50000.0])
    model = XGBRegressor(n_estimators=5, max_depth=2)
    model.fit(X, y)
    for name in ("affected_population", "fatality", "economic_damage"):
        joblib.dump(model, models_dir / f"{name}.joblib")

    scenario = DisasterScenario(
        disaster_type="Cyclone",
        region="Visakhapatnam",
        severity="Critical",
        population=186000,
        shelter_capacity=132000,
        hospital_capacity=4200,
        wind_speed=132,
        rainfall=238,
    )

    response = await orchestrator.run(scenario)

    assert response.agent_sequence == AGENT_SEQUENCE
    assert [result.agent_name for result in response.agent_results] == AGENT_SEQUENCE
    assert response.agent_results[9].agent_name == "validation"
    assert response.agent_results[10].agent_name == "planner"
    assert response.agent_results[11].agent_name == "decision"
    assert response.agent_results[12].agent_name == "reporting"
    assert response.validation_status in {"passed", "passed_with_warnings"}
    assert response.retrieved_disasters
    assert response.evidence_used


@pytest.mark.asyncio
async def test_validation_preserves_capacity_warnings_before_decision(orchestrator, monkeypatch):
    from pathlib import Path
    import joblib
    import numpy as np
    from xgboost import XGBRegressor

    models_dir = Path(__file__).resolve().parents[1] / "models" / "xgboost"
    models_dir.mkdir(parents=True, exist_ok=True)
    X = np.array([[100, 200, 3000, 10, 80, 0.5, 100000, 2000, 50000]])
    y = np.array([200000.0])
    model = XGBRegressor(n_estimators=5, max_depth=2)
    model.fit(X, y)
    for name in ("affected_population", "fatality", "economic_damage"):
        joblib.dump(model, models_dir / f"{name}.joblib")

    scenario = DisasterScenario(
        disaster_type="Flood",
        region="Low lying district",
        severity="Critical",
        population=100000,
        shelter_capacity=1000,
        hospital_capacity=300,
        wind_speed=90,
        rainfall=420,
        elevation=4,
    )

    response = await orchestrator.run(scenario)
    validation = next(r for r in response.agent_results if r.agent_name == "validation")
    decision = next(r for r in response.agent_results if r.agent_name == "decision")

    assert validation.output["validation_status"] in {"passed_with_warnings", "failed"}
    assert validation.output.get("issues") or validation.output.get("warnings")
    assert decision.output.get("unresolved_risks") is not None


@pytest.mark.asyncio
async def test_orchestration_returns_insufficient_data_without_retrieval():
    class FailingRAG:
        async def retrieve(self, scenario, top_k=10):
            return {"status": "insufficient_data", "reason": "not configured"}

    class FailingSimilarity:
        async def search_scenario(self, scenario, top_k=10):
            return {"status": "insufficient_data", "reason": "not configured"}

    orchestrator = DisasterMindOrchestrator(
        similarity_service=FailingSimilarity(),
        rag_service=FailingRAG(),
    )
    response = await orchestrator.run(
        DisasterScenario(disaster_type="Cyclone", region="Test", population=1000, shelter_capacity=500, hospital_capacity=100)
    )
    assert response.insufficient_data == {"status": "insufficient_data", "reason": "not configured"}
    assert response.agent_results == []


@pytest.mark.asyncio
async def test_orchestration_accepts_intake_after_enrichment(monkeypatch):
    async def fake_enrich(intake: ScenarioIntakeRequest) -> DisasterScenario:
        return DisasterScenario(
            disaster_type=intake.disaster_type,
            region=intake.region,
            severity=intake.severity,
            population=50000,
            hospital_capacity=2000,
            shelter_capacity=25000,
            wind_speed=80,
            rainfall=120,
            data_sources=[{"name": "test", "status": "live", "detail": "mocked"}],
        )

    monkeypatch.setattr("backend.app.services.orchestration.enrich_scenario", fake_enrich)

    response = await DisasterMindOrchestrator(
        similarity_service=MockSimilarityService(),
        rag_service=MockRAGService(),
    ).run_from_intake(
        ScenarioIntakeRequest(
            disaster_type="Flood",
            region="Chennai",
            severity="High",
        )
    )

    assert response.agent_sequence == AGENT_SEQUENCE
    knowledge = next(r for r in response.agent_results if r.agent_name == "knowledge")
    assert knowledge.output.get("historical_events")
