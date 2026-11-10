"""Task 2P Step 12 (incl. 3P): Live end-to-end verification of async RetrievalService on DEV queries, metadata filters, and no_data rejection."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env", override=False)

from backend.app.config import get_settings  # noqa: E402
from backend.app.services.embedding_service import EmbeddingService  # noqa: E402
from backend.app.services.pinecone_client import PineconeVectorClient  # noqa: E402
from backend.app.services.retrieval_service import RetrievalService  # noqa: E402

DEV_SET_PATH = PROJECT_ROOT / "eval" / "dev_set.json"
SNAPSHOT_PATH = PROJECT_ROOT / "data" / "disastermind_snapshot.parquet"
OUT_PATH = PROJECT_ROOT / "verification" / "task_2p" / "retrieval_3p_dev_eval.json"

BASELINE_METRICS = {
    "recall@1": 0.5714,
    "recall@5": 0.6190,
    "recall@10": 0.7143,
    "mrr": 0.6012,
}


def _is_relevant_meta(match: dict[str, Any], q: dict[str, Any]) -> bool:
    exp_type = (q.get("type") or "").strip().lower()
    exp_country = (q.get("country") or "").strip().lower()
    rtype = (match.get("disaster_type") or "").lower()
    rsub = (match.get("disaster_subtype") or "").lower()
    rcountry = (match.get("country") or "").lower()
    if not (exp_type in rtype or exp_type in rsub):
        return False
    if exp_country:
        return exp_country == rcountry or exp_country in rcountry
    return True


def _bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def _wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z / denom) * ((p * (1.0 - p) / n + (z * z) / (4.0 * n * n)) ** 0.5)
    return [round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)]


async def main() -> None:
    settings = get_settings()
    dev_data = json.loads(DEV_SET_PATH.read_text(encoding="utf-8"))
    queries = dev_data["retrieval_queries"]
    exp_neg_path = PROJECT_ROOT / "eval" / "dev_negatives_expanded.json"
    exp_negatives = (
        json.loads(exp_neg_path.read_text(encoding="utf-8"))["queries"]
        if exp_neg_path.exists()
        else []
    )
    snap_meta = pq.read_table(SNAPSHOT_PATH).schema.metadata or {}
    dataset_version = snap_meta.get(b"dataset_version", b"").decode("utf-8")

    emb_svc = EmbeddingService(settings=settings)
    pc_svc = PineconeVectorClient(settings=settings)
    retrieval = RetrievalService(settings=settings, embedding_service=emb_svc, pinecone_client=pc_svc)

    try:
        ret_rr: list[float] = []
        ret_r1: list[int] = []
        ret_r5: list[int] = []
        ret_r10: list[int] = []

        e2e_rr: list[float] = []
        e2e_r1: list[int] = []
        e2e_r5: list[int] = []
        e2e_r10: list[int] = []

        rejected_answerable: list[dict[str, Any]] = []
        ood_results: list[dict[str, Any]] = []
        per_query: list[dict[str, Any]] = []
        latencies_ms: list[float] = []

        # Combine 40 dev_set queries + 30 expanded DEV negatives
        eval_items = list(queries) + [
            {"id": item["id"], "query": item["query"], "category": f"neg:{item['category']}"}
            for item in exp_negatives
        ]

        for q in eval_items:
            is_negative = q["category"] == "out_of_domain" or q["category"].startswith("neg:")
            t0 = time.perf_counter()
            res = await retrieval.search(
                q["query"],
                top_k=10,
                similarity_threshold=0.0 if not is_negative else None,
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dt_ms)

            # Assert descending raw similarity scores
            sims = [m.similarity for m in res.matches]
            assert sims == sorted(sims, reverse=True), f"Non-descending scores for {q['id']}: {sims}"

            if is_negative:
                ood_results.append(
                    {
                        "id": q["id"],
                        "category": q["category"],
                        "status": res.status,
                        "top_score": res.top_score,
                        "threshold_used": res.threshold_used,
                        "returned_count": len(res.matches),
                    }
                )
                continue

            rank = None
            for idx, m in enumerate(res.matches, start=1):
                if _is_relevant_meta(m.model_dump(), q):
                    rank = idx
                    break
            rr = 1.0 / rank if rank is not None else 0.0
            ret_rr.append(rr)
            ret_r1.append(1 if rank == 1 else 0)
            ret_r5.append(1 if rank is not None and rank <= 5 else 0)
            ret_r10.append(1 if rank is not None and rank <= 10 else 0)

            abstained = (res.top_score or 0.0) < settings.similarity_threshold
            if abstained:
                rejected_answerable.append(
                    {
                        "id": q["id"],
                        "category": q["category"],
                        "query": q["query"],
                        "top_score": res.top_score,
                        "unthresholded_rank": rank,
                        "unthresholded_rr": round(rr, 4),
                    }
                )
                e2e_rr.append(0.0)
                e2e_r1.append(0)
                e2e_r5.append(0)
                e2e_r10.append(0)
            else:
                e2e_rr.append(rr)
                e2e_r1.append(1 if rank == 1 else 0)
                e2e_r5.append(1 if rank is not None and rank <= 5 else 0)
                e2e_r10.append(1 if rank is not None and rank <= 10 else 0)

            per_query.append(
                {
                    "id": q["id"],
                    "category": q["category"],
                    "rank": rank,
                    "rr": round(rr, 4),
                    "e2e_status": "no_data" if abstained else "ok",
                    "e2e_rr": 0.0 if abstained else round(rr, 4),
                    "top_score": res.top_score,
                    "above_calibrated_threshold": not abstained,
                    "record_ids_top3": res.record_ids[:3],
                    "latency_ms": round(dt_ms, 2),
                }
            )

        # Metadata filter check (hazard + country + year range)
        filtered = await retrieval.search(
            "Tropical cyclone hitting Odisha coast in India with high wind speed and storm surge",
            top_k=5,
            disaster_type="Storm",
            country="India",
            year_min=2010,
            year_max=2024,
        )
        assert filtered.status == "ok" and len(filtered.matches) > 0, f"Filtered search failed: {filtered}"
        for m in filtered.matches:
            assert m.disaster_type == "Storm"
            assert m.country == "India"
            assert m.start_year is not None and 2010 <= m.start_year <= 2024

        # Strict metadata filter returning 0 Pinecone matches triggers explicit no_data
        empty_filter_res = await retrieval.search(
            "Volcanic eruption in India",
            top_k=5,
            disaster_type="Volcanic activity",
            country="India",
            year_min=2020,
            year_max=2024,
        )
        assert empty_filter_res.status == "no_data", f"Expected no_data on infeasible filter, got {empty_filter_res.status}"

        # Explicit no_data control (unrelated query)
        no_data_res = await retrieval.search("chocolate cake baking recipe instructions", top_k=5)
        assert no_data_res.status == "no_data", f"Expected no_data, got {no_data_res.status} (top_score={no_data_res.top_score})"

        n_ans = len(ret_rr)
        rej_n = len(rejected_answerable)
        report = {
            "disclosure": "IN-SAMPLE OPTIMISTIC DEV EVALUATION (R10): DEV queries were used for model selection and negative queries for threshold calibration.",
            "dataset_version": dataset_version,
            "model": settings.embedding_model,
            "provider": emb_svc.provider,
            "dimension": settings.embedding_dimension,
            "pinecone_index_name": settings.pinecone_index_name,
            "pinecone_namespace": settings.pinecone_namespace,
            "calibrated_threshold": settings.similarity_threshold,
            "dev_answerable_count": n_ans,
            "dev_negative_count": len(ood_results),
            "retrieval_only_metrics": {
                "scope": "Unthresholded ranking over all 35 answerable DEV queries (ignores calibrated threshold)",
                "recall@1": round(float(np.mean(ret_r1)), 4),
                "recall@5": round(float(np.mean(ret_r5)), 4),
                "recall@10": round(float(np.mean(ret_r10)), 4),
                "mrr": round(float(np.mean(ret_rr)), 4),
                "mrr_ci_95": _bootstrap_ci(np.asarray(ret_rr, dtype=np.float64)),
            },
            "end_to_end_serving_metrics": {
                "scope": f"Threshold-enforced serving (threshold={settings.similarity_threshold}; rejected answerable queries count as abstentions/misses)",
                "recall@1": round(float(np.mean(e2e_r1)), 4),
                "recall@5": round(float(np.mean(e2e_r5)), 4),
                "recall@10": round(float(np.mean(e2e_r10)), 4),
                "mrr": round(float(np.mean(e2e_rr)), 4),
                "mrr_ci_95": _bootstrap_ci(np.asarray(e2e_rr, dtype=np.float64)),
                "abstention_count": rej_n,
                "abstention_rate": round(rej_n / n_ans, 4),
                "abstention_rate_wilson_95_ci": _wilson_ci(rej_n, n_ans),
                "negative_false_positives": sum(1 for r in ood_results if r["status"] == "ok"),
                "negative_total": len(ood_results),
            },
            "latency_ms": {
                "mean": round(float(np.mean(latencies_ms)), 2),
                "p95": round(float(np.percentile(latencies_ms, 95)), 2),
            },
            "rejected_answerable_queries": rejected_answerable,
            "baseline_metrics": BASELINE_METRICS,
            "metadata_filter_check": {
                "passed": True,
                "filters": filtered.filters_applied,
                "returned_count": len(filtered.matches),
                "record_ids": filtered.record_ids,
                "top_score": filtered.top_score,
            },
            "no_data_check": {
                "passed": True,
                "unrelated_query_status": no_data_res.status,
                "unrelated_top_score": no_data_res.top_score,
                "infeasible_filter_status": empty_filter_res.status,
                "threshold_used": no_data_res.threshold_used,
            },
            "ood_results": ood_results,
            "per_query": per_query,
        }
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "per_query"}, indent=2))
    finally:
        await pc_svc.aclose()
        await emb_svc.aclose()


if __name__ == "__main__":
    asyncio.run(main())

