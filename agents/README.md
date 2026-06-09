# DisasterMind Agent Directory

This folder contains the canonical agent descriptions for the DisasterMind AI multi-agent workflow.

## Execution Order

1. Coordinator
2. Knowledge (Supabase pgvector retrieval)
3. Similarity (ranked vector matches)
4. Risk (transparent contributors)
5. Prediction (XGBoost models)
6. Explainability
7. Logistics
8. Evacuation
9. Simulation
10. Validation
11. Planner
12. Decision
13. Reporting

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

