# DisasterMind AI — Architecture Audit Report

**Date:** 2026-06-09  
**Auditor role:** Lead AI Architect  
**Repository:** `Disastermind-Agent`

---

## Executive Summary

DisasterMind AI is currently a **deterministic demo orchestrator** with a Next.js dashboard. It simulates multi-agent disaster response using hardcoded formulas, fabricated confidence scores, and no connection to the 16,834-record historical dataset. The repository does **not** yet implement embeddings, pgvector retrieval, trained ML models, or evidence-backed reasoning required for the Microsoft Agents League Hackathon.

This audit documents the as-is state, every identified hardcoded value, and the gap analysis against the target real multi-agent reasoning platform.

---

## Repository Inventory

| Area | Files | Status |
|------|-------|--------|
| `data/disastermind_curated_dataset.json` | 16,834 EM-DAT-style records | **Present, unused by agents** |
| `agents/` | 11 `description.md` only | **Spec only, no Python implementations** |
| `backend/app/services/orchestration.py` | 1,090 lines | **All agent logic inline, deterministic** |
| `backend/app/services/data_enrichment.py` | Live geocoding/weather/OSM | **Real external APIs, heuristic fallbacks** |
| `embeddings/` | — | **Missing** |
| `database/` | — | **Missing** |
| `models/` | — | **Missing** |
| Supabase / pgvector | — | **Not configured** |

---

## Current Architecture (As-Is)

```
User (Next.js)
    → /api/orchestrate → FastAPI /orchestrate-intake
        → data_enrichment.enrich_scenario()  [Open-Meteo, OSM]
        → DisasterMindOrchestrator.run()
            → 11 inline _run_* methods (deterministic math)
            → optional NVIDIA LLM for reporting only
    → Dashboard renders orchestration JSON
```

**No vector search. No dataset queries. No XGBoost inference. No Supabase.**

---

## Dataset Analysis

- **Path:** `data/disastermind_curated_dataset.json`
- **Records:** 16,834
- **Key fields:** `disaster_id`, `disaster_type`, `disaster_subtype`, `event_name`, `country`, `region`, `location`, `start_year`, `total_deaths`, `total_affected`, `total_damage_usd_thousands`, `latitude`, `longitude`, `magnitude`
- **Usage today:** None in backend or frontend
- **NaN values:** Present in deaths, damage, coordinates (requires sanitization for ML)

---

## Hardcoded / Fabricated Values (Critical)

### `backend/app/schemas.py` — Default scenario (demo preset)

| Field | Hardcoded Default |
|-------|-------------------|
| `disaster_type` | `"Cyclone"` |
| `region` | `"Visakhapatnam coastal zone"` |
| `severity` | `"Critical"` |
| `population` | `186000` |
| `hospital_capacity` | `4200` |
| `shelter_capacity` | `132000` |
| `wind_speed` | `132` |
| `rainfall` | `238` |
| `elevation` | `8` |
| `population_density` | `6100` |
| `historical_damage` | `0.55` |

### `backend/app/services/orchestration.py`

| Issue | Location | Detail |
|-------|----------|--------|
| Fake confidence | Every `_run_*` agent | Static values: 0.96, 0.78, 0.86, 0.82, etc. |
| `SEVERITY_FACTOR` | Lines 38–43 | Fixed multipliers, not data-derived |
| Risk formula | `_run_risk` | Static weights: `/260`, `/600`, `/15000` |
| Prediction | `_run_prediction` | Claims XGBoost, uses `local-deterministic-v0` |
| Logistics | `_run_logistics` | Arbitrary divisors: `/18`, `/9000`, `/140` |
| Available resources | `_run_logistics` | Fake scarcity: `* 0.82`, `* 0.88`, etc. |
| Evacuation phases | `_run_evacuation` | Fixed 42%/36% split |
| Simulation | `_run_simulation` | Template scenarios with static deltas |
| Intelligence | `_run_intelligence` | `historical_events: []`, Foundry IQ pending |
| Coordinator | `_run_coordinator` | `confidence: 0.96` with no computation |

### `backend/app/services/data_enrichment.py`

| Issue | Detail |
|-------|--------|
| `_fallback_population` | Heuristic from region string length |
| `hazard_wind` / `hazard_rain` | Disaster-type formulas, not historical |
| `historical_damage` | `0.22 + severity_scale * 0.28` — not from dataset |
| `density` | `population / 42` arbitrary |

### `frontend/app/page.tsx`

| Issue | Detail |
|-------|--------|
| `buildRiskTrend` | Synthetic chart from risk/damage scores + index offsets |
| Risk map | CSS decorative zones, not geospatial data |
| Agent animation | Fake 130ms per-agent progress before API returns |
| `availableResources` fallback | `required * 0.8` when missing |

---

## Agent Directory vs Implementation

| Agent (spec) | Python module | Retrieval | Evidence |
|--------------|---------------|-----------|----------|
| coordinator | Inline | None | Scenario only |
| intelligence | Inline | None | Empty historical_events |
| risk | Inline | None | Formula only |
| prediction | Inline | None | No model artifact |
| logistics | Inline | None | Heuristic formulas |
| evacuation | Inline | None | Template zones |
| simulation | Inline | None | Template scenarios |
| validation | Inline | None | Bounds checks only |
| planner | Inline | None | Generic phases |
| decision | Inline | None | Generic actions |
| reporting | NVIDIA/local | None | No disaster citations |

**Missing per target spec:** knowledge_agent, similarity_agent, explainability_agent, real prediction models, Supabase pgvector.

---

## External Integrations (As-Is)

| Service | Used | Purpose |
|---------|------|---------|
| NVIDIA Build API | Optional | Report/answer generation |
| Open-Meteo Geocoding | Yes | Region resolution |
| Open-Meteo Forecast | Yes | Weather |
| Open-Meteo Elevation | Yes | Elevation |
| OpenStreetMap Overpass | Yes | Hospitals/shelters |
| Foundry IQ | No | Referenced as pending |
| OpenAI Embeddings | No | Not configured |
| Supabase pgvector | No | Not configured |
| PostgreSQL / Redis | No | README only |

---

## Configuration Gaps

`.env` contains NVIDIA keys only. Missing:

- `OPENAI_API_KEY` (required for `text-embedding-3-small`)
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY` or `DATABASE_URL`
- Model artifact paths

---

## Test Coverage

- `tests/test_orchestration.py` — Validates agent order and warning propagation against **deterministic** orchestrator
- `tests/test_nvidia_generation.py` — Mocks NVIDIA client
- **No tests** for vector search, RAG, ML inference, or dataset integrity

---

## Gap Analysis vs Target Platform

| Requirement | Current | Target |
|-------------|---------|--------|
| Historical evidence | None | 16,834 records via pgvector |
| Similarity scores | N/A | pgvector cosine similarity |
| Predictions | Heuristic | XGBoost on real targets |
| Risk contributors | Hidden formula | Transparent breakdown |
| Confidence | Fabricated | Derived from evidence quality |
| Recommendations | Generic templates | Cite retrieved disasters |
| Dashboard metrics | Mixed real/heuristic | 100% traceable |
| insufficient_data | Not implemented | Required when no evidence |

---

## Recommended Target Architecture

```
Dataset (16,834 records)
    → Embedding Pipeline (text-embedding-3-small)
    → Supabase disaster_embeddings (pgvector, IVFFLAT)
        → similarity_service / rag_service
            → knowledge_agent, similarity_agent
                → risk_agent (historical severity from matches)
                → prediction_agent (XGBoost artifacts)
                → explainability_agent
                → logistics / evacuation / simulation
                → validation_agent
                → planning_agent
                → decision_agent
                → reporting_agent (citations required)
    → Dashboard Evidence Panel
```

---

## Risk Register

1. **No Supabase credentials** — Blocks production retrieval until configured
2. **Embedding cost** — ~16,834 OpenAI embedding calls required once
3. **NaN in dataset** — Must sanitize for XGBoost training
4. **Orchestrator monolith** — Must refactor without breaking API contract
5. **Fake UI charts** — Risk trend and map are decorative

---

## Conclusion

The repository is a **well-structured hackathon scaffold** with real data enrichment APIs and a polished dashboard, but intelligence is **simulated**. Every agent output is computed from static formulas with invented confidence. The curated dataset is the single most valuable asset and is completely disconnected.

**Transformation priority:**

1. Embedding pipeline + Supabase pgvector
2. similarity_service + rag_service (no local fallback)
3. Modular evidence-backed agents
4. XGBoost training on historical records
5. Orchestrator refactor + dashboard evidence panel
6. Remove all fabricated metrics and template outputs

---

*End of audit report.*
