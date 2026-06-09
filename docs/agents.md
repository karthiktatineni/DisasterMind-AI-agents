# DisasterMind AI Agent Architecture

Version 1.1

---

# Agent Hierarchy

Chief Coordinator Agent
├── Intelligence Agent
├── Risk Assessment Agent
├── Prediction Agent
├── Logistics Agent
├── Evacuation Agent
├── Simulation Agent
├── Validation Agent
├── Planner Agent
├── Decision Agent
└── Reporting Agent

---

# Canonical Agent Flow

User
↓
Coordinator
↓
Intelligence
↓
Risk
↓
Prediction
↓
Logistics
↓
Evacuation
↓
Simulation
↓
Validation
↓
Planner
↓
Decision
↓
Reporting

---

# Agent Folder Structure

Each agent has a dedicated folder under `agents/` with a separate Markdown description:

```text
agents/
├── coordinator/description.md
├── intelligence/description.md
├── risk/description.md
├── prediction/description.md
├── logistics/description.md
├── evacuation/description.md
├── simulation/description.md
├── validation/description.md
├── planner/description.md
├── decision/description.md
└── reporting/description.md
```

---

# 1. Chief Coordinator Agent

Purpose:
Orchestrates the complete disaster-response workflow.

Responsibilities:

* Understand the user scenario
* Create the execution plan
* Launch agents in the required sequence
* Enforce structured input and output contracts
* Retry failed agents
* Merge validated outputs for downstream agents

Model:
GPT-4.1

Inputs:

* User scenario
* Request metadata
* Available data-source configuration

Outputs:

* Agent tasks
* Ordered execution plan
* Request-level trace identifier

---

# 2. Intelligence Agent

Purpose:
Collect grounded disaster intelligence.

Sources:

* Foundry IQ
* Disaster policies
* Historical events
* Medical response procedures
* Evacuation guidelines

Outputs:

* Grounded disaster context
* Evidence references
* Source citations
* Data gaps

Tools:

* Foundry Search
* Knowledge Retrieval
* Policy Retrieval

---

# 3. Risk Assessment Agent

Purpose:
Estimate disaster severity and exposure.

Inputs:

* Weather
* Population
* Geography
* Infrastructure exposure
* Intelligence Agent evidence package

Outputs:

```json
{
  "risk_score": 0.0,
  "affected_population": 0,
  "critical_zones": [],
  "risk_drivers": []
}
```

---

# 4. Prediction Agent

Purpose:
Run predictive ML models.

Model:
XGBoost

Inputs:

* Wind speed
* Rainfall
* Population density
* Elevation
* Historical damage
* Temperature
* Humidity
* Risk Assessment Agent output

Outputs:

```json
{
  "damage_score": 0.0,
  "casualty_estimate": 0,
  "evacuation_need": 0,
  "prediction_confidence": 0.0
}
```

---

# 5. Logistics Agent

Purpose:
Determine emergency resources required and available deployment capacity.

Calculates:

* Ambulances
* Rescue teams
* Medical supplies
* Food supplies
* Water supplies
* Temporary shelter support
* Personnel staging

Inputs:

* Prediction Agent output
* Risk Assessment Agent output
* Inventory and resource constraints
* Hospital and shelter capacity

Outputs:

* Resource deployment plan
* Resource shortfall analysis
* Staging locations
* Logistics confidence score

---

# 6. Evacuation Agent

Purpose:
Generate evacuation strategy.

Inputs:

* Population
* Roads
* Shelters
* Critical zones
* Logistics constraints

Outputs:

* Evacuation routes
* Shelter assignments
* Zone prioritization
* Traffic and transport recommendations

---

# 7. Simulation Agent

Purpose:
Run scenario simulations and stress-test response plans.

Scenarios:

* Road failure
* Shelter overload
* Hospital outage
* Flood expansion
* Cyclone escalation
* Earthquake aftershock
* Power grid failure
* Communication failure

Outputs:

* Updated plans
* Scenario deltas
* Resilience gaps
* Recommended mitigations

---

# 8. Validation Agent

Purpose:
Validate that generated plans are internally consistent, feasible, and grounded in evidence.

Responsibilities:

* Detect contradictions across agent outputs
* Verify evacuation capacity
* Verify shelter capacity
* Verify hospital capacity
* Verify logistics calculations
* Verify prediction consistency
* Reject impossible plans
* Generate validation reports

Inputs:

* Intelligence output
* Risk output
* Prediction output
* Logistics output
* Evacuation output
* Simulation output

Outputs:

```json
{
  "validation_status": "passed",
  "blocking_issues": [],
  "warnings": [],
  "required_revisions": [],
  "confidence": 0.0
}
```

---

# 9. Planner Agent

Purpose:
Convert validated agent outputs into an operational emergency plan.

Responsibilities:

* Sequence actions by urgency
* Convert recommendations into operational phases
* Assign owners, dependencies, and timing
* Prepare alternatives for degraded infrastructure
* Preserve validation constraints before decision synthesis

Inputs:

* Validation Agent report
* Simulation Agent scenarios
* Evacuation strategy
* Logistics deployment plan
* Risk and prediction outputs

Outputs:

* Operational response plan
* Phase-by-phase action timeline
* Dependency map
* Escalation triggers
* Contingency plans

---

# 10. Decision Agent

Purpose:
Create final recommendation from the validated operational plan.

Inputs:

* Planner Agent output
* Validation Agent output
* Outputs from all upstream agents

Outputs:

* Executive recommendation
* Emergency Action Plan
* Confidence score
* Explanation chain
* Decision trace

---

# 11. Reporting Agent

Purpose:
Generate structured, exportable reports for emergency operators and auditors.

Responsibilities:

* Produce executive summaries
* Produce resource deployment reports
* Produce evacuation reports
* Produce simulation reports
* Produce decision traces
* Produce audit trails
* Export reports as Markdown, JSON, and PDF

Inputs:

* Decision Agent output
* Planner Agent output
* Validation Agent report
* Evidence citations
* Audit metadata

Outputs:

* Final user-facing report
* Machine-readable JSON report
* Export package metadata

---

# Agent Communication

User
↓
Coordinator
↓
Intelligence
↓
Risk
↓
Prediction
↓
Logistics
↓
Evacuation
↓
Simulation
↓
Validation
↓
Planner
↓
Decision
↓
Reporting

Agents pass structured outputs only.

Every agent response must include:

```json
{
  "request_id": "",
  "agent_name": "",
  "timestamp": "",
  "confidence": 0.0,
  "reasoning_summary": "",
  "evidence": [],
  "recommendations": [],
  "next_actions": [],
  "status": ""
}
```

---

# Agent Memory

Short-Term:
Redis

Long-Term:
PostgreSQL

Knowledge:
Foundry IQ

---

# Agent Safety

* No hallucinated recommendations
* Must cite evidence
* Must show confidence
* Must explain reasoning
* Must validate feasibility before final decisions
* Must produce auditable reports

---

# Failure Handling

If an agent fails:

Coordinator retries.

If retry fails:

Fallback response generated.

All failures logged.
