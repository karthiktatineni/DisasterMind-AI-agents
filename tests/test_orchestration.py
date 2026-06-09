from __future__ import annotations

import pytest

from backend.app.schemas import DisasterScenario
from backend.app.schemas import ScenarioIntakeRequest
from backend.app.services.orchestration import AGENT_SEQUENCE, DisasterMindOrchestrator


@pytest.mark.asyncio
async def test_orchestration_executes_all_agents_in_required_order():
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

    response = await DisasterMindOrchestrator().run(scenario)

    assert response.agent_sequence == AGENT_SEQUENCE
    assert [result.agent_name for result in response.agent_results] == AGENT_SEQUENCE
    assert response.agent_results[7].agent_name == "validation"
    assert response.agent_results[8].agent_name == "planner"
    assert response.agent_results[9].agent_name == "decision"
    assert response.agent_results[10].agent_name == "reporting"
    assert response.validation_status in {"passed", "passed_with_warnings"}
    assert response.final_report


@pytest.mark.asyncio
async def test_validation_preserves_capacity_warnings_before_decision():
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

    response = await DisasterMindOrchestrator().run(scenario)
    validation = response.agent_results[7]
    decision = response.agent_results[9]

    assert validation.agent_name == "validation"
    assert validation.output["validation_status"] == "passed_with_warnings"
    assert any("Shelter capacity gap" in warning for warning in response.warnings)
    assert decision.output["unresolved_risks"]
    assert response.status == "completed_with_warnings"


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

    monkeypatch.setattr(
        "backend.app.services.orchestration.enrich_scenario",
        fake_enrich,
    )

    response = await DisasterMindOrchestrator().run_from_intake(
        ScenarioIntakeRequest(
            disaster_type="Flood",
            region="Chennai",
            severity="High",
        )
    )

    assert [result.agent_name for result in response.agent_results] == AGENT_SEQUENCE
    intelligence = response.agent_results[1]
    assert intelligence.output["data_sources"][0]["status"] == "live"
