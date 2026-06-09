from __future__ import annotations

from datetime import UTC, datetime
from math import ceil
from typing import Any

from backend.app.schemas import (
    AgentResult,
    AgentQuestionResponse,
    DisasterScenario,
    GenerationMessage,
    OrchestrationResponse,
    ScenarioIntakeRequest,
)
from backend.app.services.data_enrichment import enrich_scenario
from backend.app.services.nvidia_generation import (
    NvidiaBuildGenerationClient,
    NvidiaGenerationConfigError,
    NvidiaGenerationError,
)


AGENT_SEQUENCE = [
    "coordinator",
    "intelligence",
    "risk",
    "prediction",
    "logistics",
    "evacuation",
    "simulation",
    "validation",
    "planner",
    "decision",
    "reporting",
]


SEVERITY_FACTOR = {
    "Low": 0.18,
    "Moderate": 0.34,
    "High": 0.56,
    "Critical": 0.74,
}


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _evidence(source: str, title: str, detail: str) -> dict[str, str]:
    return {"source": source, "title": title, "detail": detail}


def _agent_result(
    scenario: DisasterScenario,
    agent_name: str,
    confidence: float,
    reasoning_summary: str,
    evidence: list[dict[str, Any]],
    recommendations: list[str],
    next_actions: list[str],
    status: AgentResult.model_fields["status"].annotation,
    output: dict[str, Any],
) -> AgentResult:
    return AgentResult(
        request_id=scenario.request_id,
        agent_name=agent_name,
        timestamp=_timestamp(),
        confidence=max(0, min(1, confidence)),
        reasoning_summary=reasoning_summary,
        evidence=evidence,
        recommendations=recommendations,
        next_actions=next_actions,
        status=status,
        output=output,
    )


class DisasterMindOrchestrator:
    def __init__(self, generation_client: NvidiaBuildGenerationClient | None = None):
        self.generation_client = generation_client

    async def run(self, scenario: DisasterScenario) -> OrchestrationResponse:
        results: list[AgentResult] = []

        coordinator = self._run_coordinator(scenario)
        results.append(coordinator)

        intelligence = self._run_intelligence(scenario, coordinator)
        results.append(intelligence)

        risk = self._run_risk(scenario, intelligence)
        results.append(risk)

        prediction = self._run_prediction(scenario, risk)
        results.append(prediction)

        logistics = self._run_logistics(scenario, prediction, risk)
        results.append(logistics)

        evacuation = self._run_evacuation(scenario, logistics, risk, prediction)
        results.append(evacuation)

        simulation = self._run_simulation(scenario, evacuation, logistics)
        results.append(simulation)

        validation = self._run_validation(
            scenario,
            intelligence,
            risk,
            prediction,
            logistics,
            evacuation,
            simulation,
        )
        results.append(validation)

        planner = self._run_planner(
            scenario,
            validation,
            simulation,
            logistics,
            evacuation,
        )
        results.append(planner)

        decision = self._run_decision(
            scenario,
            validation,
            planner,
            prediction,
            logistics,
            evacuation,
        )
        results.append(decision)

        report_text, provider, model = await self._run_reporting_generation(
            scenario,
            results,
            validation,
            decision,
        )

        reporting = self._run_reporting(
            scenario,
            validation,
            decision,
            report_text,
            provider,
        )
        results.append(reporting)

        warnings = list(validation.output.get("warnings", []))
        status = (
            "failed"
            if validation.output["validation_status"] == "failed"
            else "completed_with_warnings"
            if warnings
            else "completed"
        )
        confidence = round(
            sum(result.confidence for result in results) / len(results),
            3,
        )

        return OrchestrationResponse(
            request_id=scenario.request_id,
            status=status,
            agent_sequence=AGENT_SEQUENCE,
            agent_results=results,
            final_report=report_text,
            validation_status=validation.output["validation_status"],
            confidence=confidence,
            provider=provider,
            model=model,
            warnings=warnings,
        )

    async def run_from_intake(self, intake: ScenarioIntakeRequest) -> OrchestrationResponse:
        scenario = await enrich_scenario(intake)
        return await self.run(scenario)

    async def answer_question(
        self,
        scenario: DisasterScenario,
        question: str,
    ) -> AgentQuestionResponse:
        orchestration = await self.run(scenario)
        cited_agents = [
            "risk",
            "prediction",
            "logistics",
            "evacuation",
            "simulation",
            "validation",
            "decision",
        ]
        context = {
            result.agent_name: {
                "confidence": result.confidence,
                "status": result.status,
                "recommendations": result.recommendations,
                "output": result.output,
            }
            for result in orchestration.agent_results
            if result.agent_name in cited_agents
        }

        try:
            client = self.generation_client or NvidiaBuildGenerationClient()
            generated = await client.generate(
                messages=[
                    GenerationMessage(
                        role="system",
                        content=(
                            "You are DisasterMind's real-time agent answer service. "
                            "Answer only from the provided structured agent outputs. "
                            "Use computed numbers and validation warnings. Do not repeat "
                            "the user's input as the answer. If a value is unavailable, say so."
                        ),
                    ),
                    GenerationMessage(
                        role="user",
                        content=(
                            f"Question: {question}\n\n"
                            f"Scenario: {scenario.model_dump()}\n\n"
                            f"Computed agent outputs: {context}\n\n"
                            "Return: direct answer, operational implication, cited agents."
                        ),
                    ),
                ],
                max_tokens=700,
                temperature=0.15,
            )
            return AgentQuestionResponse(
                request_id=orchestration.request_id,
                answer=generated.content,
                provider=generated.provider,
                model=generated.model,
                cited_agents=cited_agents,
                confidence=orchestration.confidence,
            )
        except (NvidiaGenerationConfigError, NvidiaGenerationError):
            return AgentQuestionResponse(
                request_id=orchestration.request_id,
                answer=self._fallback_answer(question, orchestration),
                provider="local-fallback",
                model=None,
                cited_agents=cited_agents,
                confidence=orchestration.confidence,
            )

    async def answer_question_from_intake(
        self,
        intake: ScenarioIntakeRequest,
        question: str,
    ) -> AgentQuestionResponse:
        scenario = await enrich_scenario(intake)
        return await self.answer_question(scenario, question)

    def _run_coordinator(self, scenario: DisasterScenario) -> AgentResult:
        return _agent_result(
            scenario=scenario,
            agent_name="coordinator",
            confidence=0.96,
            reasoning_summary=(
                "Created the mandatory hierarchical workflow and assigned the "
                "scenario to all specialized agents in the approved order."
            ),
            evidence=[
                _evidence(
                    "scenario_input",
                    "User disaster scenario",
                    f"{scenario.disaster_type} in {scenario.region}",
                )
            ],
            recommendations=[
                "Execute every mandatory agent before issuing the final report.",
                "Preserve structured JSON outputs throughout the workflow.",
            ],
            next_actions=["Collect grounded intelligence for the scenario."],
            status="completed",
            output={
                "execution_plan": [
                    {"step": index + 1, "agent": agent}
                    for index, agent in enumerate(AGENT_SEQUENCE)
                ],
                "agent_sequence": AGENT_SEQUENCE,
                "handoff_to": "intelligence",
            },
        )

    def _run_intelligence(
        self,
        scenario: DisasterScenario,
        coordinator: AgentResult,
    ) -> AgentResult:
        citations = [
            {
                "id": "scenario_input",
                "label": "Operator-provided scenario facts",
                "scope": "disaster type, region, capacities, weather indicators",
            },
            {
                "id": "foundry_iq_pending",
                "label": "Foundry IQ retrieval hook",
                "scope": "protocol retrieval pending live knowledge configuration",
            },
        ]
        confidence = 0.78 if scenario.notes else 0.72
        return _agent_result(
            scenario=scenario,
            agent_name="intelligence",
            confidence=confidence,
            reasoning_summary=(
                "Built a grounded context package from scenario facts and marked "
                "external policy retrieval as pending when no Foundry IQ source is configured."
            ),
            evidence=[
                _evidence(
                    "scenario_input",
                    "Weather and capacity indicators",
                    (
                        f"Wind {scenario.wind_speed} km/h, rainfall {scenario.rainfall} mm, "
                        f"population {scenario.population}"
                    ),
                )
            ],
            recommendations=[
                "Use operator-provided facts as primary evidence for this run.",
                "Attach Foundry IQ policies when the knowledge source is configured.",
            ],
            next_actions=["Pass context package to Risk Agent."],
            status="completed",
            output={
                "grounded_context": {
                    "hazard": scenario.disaster_type,
                    "region": scenario.region,
                    "resolved_location": scenario.resolved_location,
                    "country": scenario.country,
                    "coordinates": {
                        "latitude": scenario.latitude,
                        "longitude": scenario.longitude,
                    },
                    "time_horizon": scenario.time_horizon,
                    "operator_notes": scenario.notes,
                },
                "citations": citations,
                "data_sources": scenario.data_sources,
                "policy_references": [],
                "historical_events": [],
                "data_gaps": [
                    "Foundry IQ policy retrieval is not configured in this local build."
                ],
                "coordinator_trace": coordinator.output["execution_plan"],
            },
        )

    def _run_risk(
        self,
        scenario: DisasterScenario,
        intelligence: AgentResult,
    ) -> AgentResult:
        weather_component = min(0.42, scenario.wind_speed / 260 * 0.24 + scenario.rainfall / 600 * 0.18)
        exposure_component = min(0.28, scenario.population_density / 15000 * 0.18 + scenario.historical_damage * 0.1)
        severity_component = SEVERITY_FACTOR[scenario.severity] * 0.3
        low_elevation_component = 0.08 if scenario.elevation < 15 else 0.03
        risk_score = round(
            min(0.99, weather_component + exposure_component + severity_component + low_elevation_component),
            3,
        )
        critical_population = round(scenario.population * min(0.82, risk_score * 0.88))
        risk_level = (
            "critical"
            if risk_score >= 0.75
            else "high"
            if risk_score >= 0.55
            else "medium"
            if risk_score >= 0.35
            else "low"
        )

        return _agent_result(
            scenario=scenario,
            agent_name="risk",
            confidence=0.86,
            reasoning_summary=(
                "Combined severity, weather intensity, population density, historical damage, "
                "and elevation exposure into a normalized risk profile."
            ),
            evidence=[
                _evidence("intelligence", "Grounded context", str(intelligence.output["grounded_context"])),
                _evidence("scenario_input", "Population exposure", str(scenario.population)),
            ],
            recommendations=[
                f"Treat the scenario as {risk_level} risk.",
                "Prioritize low-elevation and high-density zones for evacuation planning.",
            ],
            next_actions=["Send normalized risk profile to Prediction Agent."],
            status="completed",
            output={
                "risk_score": risk_score,
                "risk_level": risk_level,
                "affected_population": critical_population,
                "critical_zones": [
                    {
                        "zone": "coastal_low_elevation_zone",
                        "reason": "Low elevation and high rainfall exposure",
                        "priority": 1,
                    },
                    {
                        "zone": "high_density_urban_corridor",
                        "reason": "High population density and transport pressure",
                        "priority": 2,
                    },
                ],
                "risk_drivers": [
                    "wind_speed",
                    "rainfall",
                    "population_density",
                    "elevation",
                    "historical_damage",
                ],
            },
        )

    def _run_prediction(
        self,
        scenario: DisasterScenario,
        risk: AgentResult,
    ) -> AgentResult:
        risk_score = float(risk.output["risk_score"])
        damage_score = round(
            min(
                0.99,
                risk_score * 0.54
                + scenario.historical_damage * 0.24
                + min(scenario.wind_speed / 300, 1) * 0.14
                + min(scenario.rainfall / 700, 1) * 0.08,
            ),
            3,
        )
        evacuation_need = round(
            min(
                scenario.population,
                scenario.population * (SEVERITY_FACTOR[scenario.severity] * 0.58 + risk_score * 0.34),
            )
        )
        casualty_estimate = max(
            0,
            round(evacuation_need * (0.0018 + damage_score * 0.006)),
        )

        return _agent_result(
            scenario=scenario,
            agent_name="prediction",
            confidence=0.82,
            reasoning_summary=(
                "Estimated disaster impact with an XGBoost-style feature contract. "
                "The local engine uses deterministic feature math until a trained model artifact is registered."
            ),
            evidence=[
                _evidence("risk", "Risk score", str(risk_score)),
                _evidence("scenario_input", "Model features", "wind, rainfall, density, elevation, damage history"),
            ],
            recommendations=[
                "Use evacuation demand as the sizing input for Logistics and Evacuation agents.",
                "Register trained XGBoost and LightGBM artifacts before production deployment.",
            ],
            next_actions=["Calculate resource deployment requirements."],
            status="completed",
            output={
                "model_name": "xgboost-disaster-impact",
                "model_version": "local-deterministic-v0",
                "damage_score": damage_score,
                "risk_score": risk_score,
                "casualty_estimate": casualty_estimate,
                "evacuation_need": evacuation_need,
                "feature_warnings": [
                    "No persisted XGBoost artifact registered; deterministic contract used."
                ],
            },
        )

    def _run_logistics(
        self,
        scenario: DisasterScenario,
        prediction: AgentResult,
        risk: AgentResult,
    ) -> AgentResult:
        evacuation_need = int(prediction.output["evacuation_need"])
        casualty_estimate = int(prediction.output["casualty_estimate"])
        risk_score = float(risk.output["risk_score"])
        ambulances = max(4, ceil(casualty_estimate / 18))
        rescue_teams = max(3, ceil(evacuation_need / 9000))
        medical_kits = max(100, ceil(evacuation_need / 140))
        water_units = ceil(evacuation_need * 2.8)
        food_packets = ceil(evacuation_need * 2.2)
        available_ambulances = max(1, round(ambulances * (0.82 if risk_score > 0.7 else 1.05)))
        available_rescue_teams = max(1, round(rescue_teams * 0.9))
        available_medical_kits = max(0, round(medical_kits * 0.88))
        available_water_units = max(0, round(water_units * 0.84))
        available_food_packets = max(0, round(food_packets * 0.86))

        shortfalls = []
        if available_ambulances < ambulances:
            shortfalls.append(
                {
                    "resource": "ambulances",
                    "required": ambulances,
                    "available": available_ambulances,
                    "gap": ambulances - available_ambulances,
                }
            )
        if available_rescue_teams < rescue_teams:
            shortfalls.append(
                {
                    "resource": "rescue_teams",
                    "required": rescue_teams,
                    "available": available_rescue_teams,
                    "gap": rescue_teams - available_rescue_teams,
                }
            )

        return _agent_result(
            scenario=scenario,
            agent_name="logistics",
            confidence=0.84 if not shortfalls else 0.76,
            reasoning_summary=(
                "Converted predicted evacuation and casualty demand into vehicle, team, "
                "medical, water, and food requirements with explicit shortfall checks."
            ),
            evidence=[
                _evidence("prediction", "Evacuation need", str(evacuation_need)),
                _evidence("prediction", "Casualty estimate", str(casualty_estimate)),
            ],
            recommendations=[
                "Stage resources outside flood-prone and high-wind corridors.",
                "Prioritize ambulance and rescue-team gaps before public evacuation starts.",
            ],
            next_actions=["Pass capacity and staging constraints to Evacuation Agent."],
            status="completed",
            output={
                "resource_plan": {
                    "ambulances": ambulances,
                    "rescue_teams": rescue_teams,
                    "medical_kits": medical_kits,
                    "water_units": water_units,
                    "food_packets": food_packets,
                },
                "available_resources": {
                    "ambulances": available_ambulances,
                    "rescue_teams": available_rescue_teams,
                    "medical_kits": available_medical_kits,
                    "water_units": available_water_units,
                    "food_packets": available_food_packets,
                },
                "resource_shortfalls": shortfalls,
                "staging_locations": [
                    "north_inland_staging_area",
                    "hospital_support_node",
                    "shelter_distribution_hub",
                ],
                "deployment_waves": [
                    {"wave": 1, "focus": "medical response and evacuation transport"},
                    {"wave": 2, "focus": "shelter support and food-water distribution"},
                ],
                "constraints": ["road exposure", "shelter capacity", "hospital backup power"],
                "source_scenario": {
                    "population": scenario.population,
                    "hospital_capacity": scenario.hospital_capacity,
                    "shelter_capacity": scenario.shelter_capacity,
                    "wind_speed": scenario.wind_speed,
                    "rainfall": scenario.rainfall,
                    "elevation": scenario.elevation,
                    "population_density": scenario.population_density,
                    "temperature": scenario.temperature,
                    "humidity": scenario.humidity,
                    "latitude": scenario.latitude,
                    "longitude": scenario.longitude,
                    "resolved_location": scenario.resolved_location,
                    "data_sources": scenario.data_sources,
                },
            },
        )

    def _run_evacuation(
        self,
        scenario: DisasterScenario,
        logistics: AgentResult,
        risk: AgentResult,
        prediction: AgentResult,
    ) -> AgentResult:
        evacuation_need = int(prediction.output["evacuation_need"])
        shelter_gap = max(0, evacuation_need - scenario.shelter_capacity)
        assigned_to_primary = min(evacuation_need, scenario.shelter_capacity)
        phase_one = ceil(evacuation_need * 0.42)
        phase_two = ceil(evacuation_need * 0.36)
        phase_three = max(0, evacuation_need - phase_one - phase_two)

        route_conflicts = []
        if scenario.rainfall > 180:
            route_conflicts.append("coastal_arterial_flooding_risk")
        if scenario.wind_speed > 120:
            route_conflicts.append("high_wind_vehicle_restriction")

        return _agent_result(
            scenario=scenario,
            agent_name="evacuation",
            confidence=0.8 if shelter_gap == 0 else 0.7,
            reasoning_summary=(
                "Built a phased evacuation strategy from risk zones, predicted demand, "
                "shelter capacity, and logistics constraints."
            ),
            evidence=[
                _evidence("risk", "Critical zones", str(risk.output["critical_zones"])),
                _evidence("logistics", "Resource constraints", str(logistics.output["constraints"])),
            ],
            recommendations=[
                "Evacuate critical coastal zones first.",
                "Use inland route alternatives where rainfall or wind affects arterial roads.",
            ],
            next_actions=["Stress-test routes, shelters, and hospitals in Simulation Agent."],
            status="completed",
            output={
                "evacuation_zones": [
                    {"zone": "coastal_low_elevation_zone", "phase": 1, "population": phase_one},
                    {"zone": "high_density_urban_corridor", "phase": 2, "population": phase_two},
                    {"zone": "secondary_watch_zone", "phase": 3, "population": phase_three},
                ],
                "routes": [
                    {"name": "inland_north_corridor", "status": "primary"},
                    {"name": "western_ring_route", "status": "backup"},
                    {"name": "coastal_arterial", "status": "restricted"},
                ],
                "shelter_assignments": [
                    {
                        "shelter_group": "primary_verified_shelters",
                        "assigned_population": assigned_to_primary,
                        "capacity": scenario.shelter_capacity,
                    }
                ],
                "transport_requirements": [
                    {
                        "group": "medical_and_low_mobility",
                        "priority": 1,
                        "support": "ambulance and assisted transport",
                    }
                ],
                "shelter_gap": shelter_gap,
                "route_conflicts": route_conflicts,
            },
        )

    def _run_simulation(
        self,
        scenario: DisasterScenario,
        evacuation: AgentResult,
        logistics: AgentResult,
    ) -> AgentResult:
        shelter_gap = int(evacuation.output["shelter_gap"])
        route_conflicts = list(evacuation.output["route_conflicts"])
        scenario_results = [
            {
                "scenario": "road_failure",
                "status": "warning" if route_conflicts else "passed",
                "delta": "+18 minutes route delay" if route_conflicts else "no material route delay",
            },
            {
                "scenario": "shelter_saturation",
                "status": "warning" if shelter_gap > 0 else "passed",
                "delta": f"{shelter_gap} overflow population" if shelter_gap > 0 else "capacity within limit",
            },
            {
                "scenario": "hospital_outage",
                "status": "warning" if scenario.hospital_capacity < 5000 else "passed",
                "delta": "backup referral network required",
            },
            {
                "scenario": "power_grid_failure",
                "status": "warning",
                "delta": "generator and communications redundancy required",
            },
        ]
        resilience_gaps = [
            result for result in scenario_results if result["status"] == "warning"
        ]

        return _agent_result(
            scenario=scenario,
            agent_name="simulation",
            confidence=0.79,
            reasoning_summary=(
                "Stress-tested the evacuation and logistics plans against road, shelter, hospital, "
                "and grid disruptions and recorded plan deltas."
            ),
            evidence=[
                _evidence("evacuation", "Route conflicts", str(route_conflicts)),
                _evidence("logistics", "Resource plan", str(logistics.output["resource_plan"])),
            ],
            recommendations=[
                "Prepare alternate route activation triggers.",
                "Keep overflow shelters and hospital referrals ready before impact.",
            ],
            next_actions=["Validate feasibility, capacity, and contradictions."],
            status="completed",
            output={
                "scenario_results": scenario_results,
                "plan_deltas": [result["delta"] for result in scenario_results],
                "resilience_gaps": resilience_gaps,
                "mitigations": [
                    "Activate western ring route if coastal arterial is blocked.",
                    "Open overflow shelters when verified shelter load exceeds 85%.",
                    "Pre-stage backup power at hospital and shelter nodes.",
                ],
            },
        )

    def _run_validation(
        self,
        scenario: DisasterScenario,
        intelligence: AgentResult,
        risk: AgentResult,
        prediction: AgentResult,
        logistics: AgentResult,
        evacuation: AgentResult,
        simulation: AgentResult,
    ) -> AgentResult:
        blocking_issues = []
        warnings = []
        required_revisions = []
        evacuation_need = int(prediction.output["evacuation_need"])
        affected_population = int(risk.output["affected_population"])
        shelter_gap = int(evacuation.output["shelter_gap"])

        if evacuation_need > scenario.population:
            blocking_issues.append("Evacuation need exceeds total population.")
        if affected_population > scenario.population:
            blocking_issues.append("Affected population exceeds total population.")
        if shelter_gap > 0:
            warnings.append(
                f"Shelter capacity gap of {shelter_gap} people requires overflow planning."
            )
            required_revisions.append("Activate overflow shelters or reduce exposed shelter load.")
        if logistics.output["resource_shortfalls"]:
            warnings.append("Resource shortfalls exist in logistics plan.")
        if intelligence.output["data_gaps"]:
            warnings.append("External policy grounding is pending Foundry IQ configuration.")
        if simulation.output["resilience_gaps"]:
            warnings.append("Simulation found infrastructure resilience gaps.")

        validation_status = (
            "failed"
            if blocking_issues
            else "passed_with_warnings"
            if warnings
            else "passed"
        )

        return _agent_result(
            scenario=scenario,
            agent_name="validation",
            confidence=0.9 if not blocking_issues else 0.55,
            reasoning_summary=(
                "Checked population bounds, shelter capacity, logistics shortfalls, "
                "simulation gaps, and external evidence coverage before planning."
            ),
            evidence=[
                _evidence("prediction", "Evacuation need", str(evacuation_need)),
                _evidence("evacuation", "Shelter gap", str(shelter_gap)),
                _evidence("simulation", "Resilience gaps", str(simulation.output["resilience_gaps"])),
            ],
            recommendations=[
                "Proceed to Planner with warnings preserved."
                if not blocking_issues
                else "Stop execution-ready planning until blocking issues are fixed."
            ],
            next_actions=[
                "Create operational phases with warnings included."
                if not blocking_issues
                else "Return remediation plan."
            ],
            status="completed",
            output={
                "validation_status": validation_status,
                "blocking_issues": blocking_issues,
                "warnings": warnings,
                "required_revisions": required_revisions,
                "checked_agents": [
                    intelligence.agent_name,
                    risk.agent_name,
                    prediction.agent_name,
                    logistics.agent_name,
                    evacuation.agent_name,
                    simulation.agent_name,
                ],
            },
        )

    def _run_planner(
        self,
        scenario: DisasterScenario,
        validation: AgentResult,
        simulation: AgentResult,
        logistics: AgentResult,
        evacuation: AgentResult,
    ) -> AgentResult:
        validation_failed = validation.output["validation_status"] == "failed"
        phases = (
            [
                {
                    "phase": "remediation",
                    "trigger": "validation_failed",
                    "owner": "Emergency operations chief",
                    "action": "Resolve blocking feasibility issues before field execution.",
                }
            ]
            if validation_failed
            else [
                {
                    "phase": "0-6h",
                    "trigger": "scenario_confirmed",
                    "owner": "Coordinator",
                    "action": "Activate command structure and evidence retrieval.",
                },
                {
                    "phase": "6-18h",
                    "trigger": "risk_score_high",
                    "owner": "Logistics and Evacuation leads",
                    "action": "Stage resources and begin phase-one evacuation.",
                },
                {
                    "phase": "18-48h",
                    "trigger": "shelter_or_route_warning",
                    "owner": "Planning cell",
                    "action": "Open overflow shelters and switch routes when thresholds are crossed.",
                },
            ]
        )

        return _agent_result(
            scenario=scenario,
            agent_name="planner",
            confidence=0.82 if not validation_failed else 0.58,
            reasoning_summary=(
                "Converted validated agent outputs into phased operational actions, "
                "dependencies, contingencies, and escalation triggers."
            ),
            evidence=[
                _evidence("validation", "Validation status", validation.output["validation_status"]),
                _evidence("simulation", "Mitigations", str(simulation.output["mitigations"])),
            ],
            recommendations=[
                "Use phased activation with explicit triggers.",
                "Keep validation warnings visible during execution.",
            ],
            next_actions=["Submit operational plan to Decision Agent."],
            status="completed",
            output={
                "operational_plan": {
                    "region": scenario.region,
                    "disaster_type": scenario.disaster_type,
                    "validation_status": validation.output["validation_status"],
                },
                "phases": phases,
                "dependencies": [
                    "shelter_capacity_confirmation",
                    "ambulance_gap_resolution",
                    "route_status_updates",
                ],
                "owners": [
                    "Coordinator",
                    "Logistics lead",
                    "Evacuation lead",
                    "Hospital liaison",
                    "Shelter operations lead",
                ],
                "contingencies": simulation.output["mitigations"],
                "escalation_triggers": [
                    "shelter load exceeds 85%",
                    "primary road closed",
                    "hospital backup power fails",
                ],
                "source_plans": {
                    "logistics": logistics.output["resource_plan"],
                    "evacuation": evacuation.output["evacuation_zones"],
                },
            },
        )

    def _run_decision(
        self,
        scenario: DisasterScenario,
        validation: AgentResult,
        planner: AgentResult,
        prediction: AgentResult,
        logistics: AgentResult,
        evacuation: AgentResult,
    ) -> AgentResult:
        validation_failed = validation.output["validation_status"] == "failed"
        unresolved = list(validation.output["blocking_issues"])
        if validation.output["warnings"]:
            unresolved.extend(validation.output["warnings"])
        recommendation = (
            "Human review required before execution."
            if validation_failed
            else "Proceed with phased evacuation and resource staging while resolving warnings."
        )

        return _agent_result(
            scenario=scenario,
            agent_name="decision",
            confidence=0.84 if not validation_failed else 0.5,
            reasoning_summary=(
                "Synthesized the operational plan, validation findings, prediction outputs, "
                "logistics constraints, and evacuation strategy into an auditable decision."
            ),
            evidence=[
                _evidence("planner", "Operational phases", str(planner.output["phases"])),
                _evidence("validation", "Warnings", str(validation.output["warnings"])),
                _evidence("prediction", "Evacuation need", str(prediction.output["evacuation_need"])),
            ],
            recommendations=[recommendation],
            next_actions=["Generate final executive and audit reports."],
            status="needs_human_review" if validation_failed else "completed",
            output={
                "executive_recommendation": recommendation,
                "emergency_action_plan": planner.output["operational_plan"],
                "prioritized_actions": [
                    "Activate emergency operations coordination.",
                    "Stage ambulances, rescue teams, water, food, and medical kits.",
                    "Begin phase-one evacuation for critical zones.",
                    "Open overflow shelter capacity if threshold is crossed.",
                    "Switch to alternate routes if road simulation trigger occurs.",
                ],
                "decision_trace": [
                    "Risk and prediction quantified exposure.",
                    "Logistics sized resources from predicted demand.",
                    "Evacuation assigned zones and shelter loads.",
                    "Simulation identified stress failures.",
                    "Validation preserved warnings before planning.",
                ],
                "unresolved_risks": unresolved,
                "source_outputs": {
                    "prediction": prediction.output,
                    "logistics": logistics.output,
                    "evacuation": evacuation.output,
                },
            },
        )

    async def _run_reporting_generation(
        self,
        scenario: DisasterScenario,
        results: list[AgentResult],
        validation: AgentResult,
        decision: AgentResult,
    ) -> tuple[str, str, str | None]:
        prompt = self._build_reporting_prompt(scenario, results, validation, decision)
        try:
            client = self.generation_client or NvidiaBuildGenerationClient()
            generated = await client.generate(
                messages=[
                    GenerationMessage(
                        role="system",
                        content=(
                            "You are the DisasterMind Reporting Agent. Produce a concise, "
                            "operational emergency report from structured agent outputs. "
                            "Do not invent external citations."
                        ),
                    ),
                    GenerationMessage(role="user", content=prompt),
                ],
                max_tokens=900,
                temperature=0.2,
            )
            return generated.content, generated.provider, generated.model
        except (NvidiaGenerationConfigError, NvidiaGenerationError):
            return self._fallback_report(scenario, validation, decision), "local-fallback", None

    def _run_reporting(
        self,
        scenario: DisasterScenario,
        validation: AgentResult,
        decision: AgentResult,
        report_text: str,
        provider: str,
    ) -> AgentResult:
        return _agent_result(
            scenario=scenario,
            agent_name="reporting",
            confidence=0.86 if provider == "nvidia-build" else 0.72,
            reasoning_summary=(
                "Generated final report content and preserved validation warnings, "
                "decision trace, and audit-relevant outputs."
            ),
            evidence=[
                _evidence("decision", "Executive recommendation", decision.output["executive_recommendation"]),
                _evidence("validation", "Validation status", validation.output["validation_status"]),
            ],
            recommendations=["Export Markdown, JSON, and PDF reports when file renderers are configured."],
            next_actions=["Deliver final report to the dashboard."],
            status="completed",
            output={
                "executive_summary": decision.output["executive_recommendation"],
                "reports": {
                    "emergency_action_plan": decision.output["emergency_action_plan"],
                    "decision_trace": decision.output["decision_trace"],
                    "validation": validation.output,
                },
                "export_paths": [],
                "generation_provider": provider,
                "report_text": report_text,
            },
        )

    def _build_reporting_prompt(
        self,
        scenario: DisasterScenario,
        results: list[AgentResult],
        validation: AgentResult,
        decision: AgentResult,
    ) -> str:
        compact_outputs = {
            result.agent_name: {
                "confidence": result.confidence,
                "status": result.status,
                "output": result.output,
            }
            for result in results
        }
        return (
            "Scenario:\n"
            f"- Disaster: {scenario.disaster_type}\n"
            f"- Region: {scenario.region}\n"
            f"- Time horizon: {scenario.time_horizon}\n"
            f"- Severity: {scenario.severity}\n"
            f"- Population: {scenario.population}\n\n"
            "Agent outputs:\n"
            f"{compact_outputs}\n\n"
            "Validation:\n"
            f"{validation.output}\n\n"
            "Decision:\n"
            f"{decision.output}\n\n"
            "Write sections: Executive Summary, Emergency Action Plan, Resource Deployment, "
            "Evacuation Plan, Simulation Risks, Validation Warnings, Confidence."
        )

    def _fallback_report(
        self,
        scenario: DisasterScenario,
        validation: AgentResult,
        decision: AgentResult,
    ) -> str:
        warnings = validation.output.get("warnings", [])
        warning_text = "\n".join(f"- {warning}" for warning in warnings) or "- No warnings."
        actions = "\n".join(
            f"{index + 1}. {action}"
            for index, action in enumerate(decision.output["prioritized_actions"])
        )
        return (
            "Executive Summary\n"
            f"{decision.output['executive_recommendation']} Scenario: {scenario.disaster_type} "
            f"in {scenario.region} over {scenario.time_horizon}.\n\n"
            "Emergency Action Plan\n"
            f"{actions}\n\n"
            "Validation Warnings\n"
            f"{warning_text}\n\n"
            "Confidence\n"
            "Local deterministic orchestration completed. NVIDIA reporting was not available "
            "for this request, so this fallback report preserves structured agent outputs."
        )

    def _fallback_answer(self, question: str, orchestration: OrchestrationResponse) -> str:
        outputs = {result.agent_name: result.output for result in orchestration.agent_results}
        risk = outputs.get("risk", {})
        prediction = outputs.get("prediction", {})
        logistics = outputs.get("logistics", {})
        evacuation = outputs.get("evacuation", {})
        validation = outputs.get("validation", {})
        return (
            f"Answer to: {question}\n"
            f"Risk score is {risk.get('risk_score')} with level {risk.get('risk_level')}. "
            f"Predicted evacuation need is {prediction.get('evacuation_need')} and casualty "
            f"estimate is {prediction.get('casualty_estimate')}. Logistics requires "
            f"{logistics.get('resource_plan', {})}. Evacuation shelter gap is "
            f"{evacuation.get('shelter_gap')}. Validation status is "
            f"{validation.get('validation_status')} with warnings: {validation.get('warnings', [])}."
        )
