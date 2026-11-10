#!/usr/bin/env python3
"""DisasterMind AI - Latency Benchmarking (p50, p95, p99)
Measures latency breakdown across:
1. Vector similarity search over 12,000 embedded records
2. Live external data enrichment (Open-Meteo, Nominatim, Overpass)
3. Full agent orchestration pipeline (deterministic fallback path)
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.schemas import DisasterScenario, ScenarioIntakeRequest
from backend.app.services.data_enrichment import enrich_scenario
from backend.app.services.orchestration import DisasterMindOrchestrator
try:
    from eval.audit_test_suite import ExactCosineRAGService, ExactCosineSimService
except ImportError:
    from verification.audit_test_suite import ExactCosineRAGService, ExactCosineSimService

EMBEDDINGS_OUTPUT_PATH = PROJECT_ROOT / "embeddings" / "embedded_disasters_output.json"

with open(EMBEDDINGS_OUTPUT_PATH, encoding="utf-8") as f:
    records = json.load(f)

matrix = np.array([r["embedding"] for r in records], dtype=np.float32)

rag = ExactCosineRAGService(records, matrix)
sim = ExactCosineSimService(rag)
orchestrator = DisasterMindOrchestrator(similarity_service=sim, rag_service=rag)

def stats(latencies: list[float]) -> dict[str, float]:
    arr = np.array(latencies)
    return {
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "p99": round(float(np.percentile(arr, 99)), 2),
        "mean": round(float(np.mean(arr)), 2),
        "min": round(float(np.min(arr)), 2),
        "max": round(float(np.max(arr)), 2),
    }

async def benchmark_vector_retrieval(n=50):
    print(f"Benchmarking Vector Retrieval across 12,000 records ({n} runs)...")
    latencies = []
    # simulate 50 queries
    sample_vecs = matrix[:n]
    for i in range(n):
        t0 = time.perf_counter()
        _ = rag.search_vector(sample_vecs[i], top_k=10)
        latencies.append((time.perf_counter() - t0) * 1000)
    return stats(latencies)

async def benchmark_orchestration_pipeline(n=20):
    print(f"Benchmarking End-to-End Orchestrator Pipeline ({n} runs)...")
    latencies = []
    sc = DisasterScenario(
        disaster_type="Cyclone",
        region="Visakhapatnam",
        severity="Critical",
        population=186000,
        shelter_capacity=132000,
        hospital_capacity=4200,
        wind_speed=132,
        rainfall=238,
    )
    for _ in range(n):
        t0 = time.perf_counter()
        _ = await orchestrator.run(sc)
        latencies.append((time.perf_counter() - t0) * 1000)
    return stats(latencies)

async def benchmark_enrichment(n=5):
    print(f"Benchmarking Live Data Enrichment APIs ({n} runs)...")
    latencies = []
    intake = ScenarioIntakeRequest(
        disaster_type="Flood",
        region="Chennai",
        severity="High",
    )
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            _ = await enrich_scenario(intake)
            latencies.append((time.perf_counter() - t0) * 1000)
        except Exception as e:
            print(f"  Enrichment failed: {e}")
    return stats(latencies) if latencies else {}

async def main():
    vec_stats = await benchmark_vector_retrieval()
    orch_stats = await benchmark_orchestration_pipeline()
    enrich_stats = await benchmark_enrichment()

    res = {
        "vector_search_12k_ms": vec_stats,
        "agent_pipeline_e2e_ms": orch_stats,
        "external_enrichment_ms": enrich_stats,
    }
    print("\nBENCHMARK RESULTS:")
    print(json.dumps(res, indent=2))
    with open(PROJECT_ROOT / "verification" / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main())

