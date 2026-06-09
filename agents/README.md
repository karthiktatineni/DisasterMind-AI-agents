# DisasterMind Agent Directory

This folder contains the canonical agent descriptions for the DisasterMind AI multi-agent workflow.

## Execution Order

1. Coordinator
2. Intelligence
3. Risk
4. Prediction
5. Logistics
6. Evacuation
7. Simulation
8. Validation
9. Planner
10. Decision
11. Reporting

## Folder Layout

Each agent has its own folder with a `description.md` file:

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

## Shared Communication Contract

Every agent output must be structured JSON and must include:

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

Agents must not exchange raw narrative text as their primary output. Narrative text is only allowed inside structured fields such as `reasoning_summary`, `recommendations`, and report sections.

