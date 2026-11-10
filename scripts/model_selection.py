"""Task 2P Step 8: Candidate Embedding Model Selection Benchmark on DEV Set.

Follows verification/embedding_selection_rule.md strictly:
- Never touches eval/holdout_sealed.json
- Caches embeddings locally under eval/.cache/ (gitignored)
- Evaluates exact L2-normalized cosine similarity in memory over all 16,833 snapshot records
- Reports Recall@1/5/10, MRR with 95% bootstrap CIs (2000 resamples, seed=42), paired delta CIs,
  calibrated similarity threshold, OOD false positives, latency, and dimension.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI
from pinecone import Pinecone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env", override=False)

SNAPSHOT_PATH = PROJECT_ROOT / "data" / "disastermind_snapshot.parquet"
DEV_SET_PATH = PROJECT_ROOT / "eval" / "dev_set.json"
CACHE_ROOT = PROJECT_ROOT / "eval" / ".cache"
OUT_DIR = PROJECT_ROOT / "verification" / "task_2p"

BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 42
BASELINE_MRR = 0.6012


@dataclass(frozen=True)
class CandidateModel:
    model_id: str
    provider: str  # "nvidia" or "pinecone"
    dimension: int
    batch_size: int
    concurrency: int

    @property
    def slug(self) -> str:
        return self.model_id.replace("/", "__")


CANDIDATES = [
    CandidateModel("nvidia/nemotron-3.5-embed-1b", "nvidia", 2048, 1, 32),
    CandidateModel("nvidia/nemotron-3-embed-1b", "nvidia", 2048, 50, 4),
    CandidateModel("nvidia/nemotron-3.5-embed-8b", "nvidia", 4096, 1, 32),
    CandidateModel("llama-text-embed-v2", "pinecone", 1024, 96, 1),
    CandidateModel("multilingual-e5-large", "pinecone", 1024, 96, 1),
]


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (arr / norms).astype(np.float32)


async def _embed_nvidia_batch(
    client: AsyncOpenAI,
    model_id: str,
    texts: list[str],
    input_type: str,
    sem: asyncio.Semaphore,
    max_retries: int = 6,
) -> list[list[float]]:
    async with sem:
        for attempt in range(max_retries):
            try:
                resp = await client.embeddings.create(
                    model=model_id,
                    input=texts,
                    extra_body={
                        "input_type": input_type,
                        "encoding_format": "float",
                        "truncate": "END",
                    },
                )
                return [item.embedding for item in resp.data]
            except APIStatusError as exc:
                if exc.status_code not in (429, 500, 502, 503, 504) or attempt == max_retries - 1:
                    raise
            except (APIConnectionError, APITimeoutError):
                if attempt == max_retries - 1:
                    raise
            await asyncio.sleep(1.0 * (2**attempt) + random.uniform(0.0, 0.6))
    raise RuntimeError("Unreachable")


async def _embed_nvidia_passages(
    candidate: CandidateModel,
    texts: list[str],
    ids: list[str],
    dataset_version: str,
    cache_dir: Path,
) -> np.ndarray:
    passages_npy = cache_dir / "passages.npy"
    meta_json = cache_dir / "cache_meta.json"
    if passages_npy.exists() and meta_json.exists():
        meta = json.loads(meta_json.read_text(encoding="utf-8"))
        if meta.get("dataset_version") == dataset_version and meta.get("complete") is True:
            arr = np.load(passages_npy)
            if arr.shape == (len(texts), candidate.dimension):
                print(f"  [CACHE HIT] {candidate.model_id}: loaded {arr.shape} from {passages_npy}")
                return arr

    # Resume support via partial memmap/npy or shard files
    shards_dir = cache_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    client = AsyncOpenAI(
        api_key=os.getenv("NVIDIA_API_KEY"),
        base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/"),
        timeout=120,
        max_retries=0,
    )
    sem = asyncio.Semaphore(candidate.concurrency)
    bs = candidate.batch_size
    shard_size = 200 if bs == 1 else bs
    total_shards = (len(texts) + shard_size - 1) // shard_size

    async def process_shard(s_idx: int) -> None:
        shard_path = shards_dir / f"batch_{s_idx:04d}.npy"
        if shard_path.exists():
            return
        start = s_idx * shard_size
        end = min(len(texts), start + shard_size)
        chunk = texts[start:end]
        if bs == 1:
            sub_res = await asyncio.gather(
                *(_embed_nvidia_batch(client, candidate.model_id, [t], "passage", sem, max_retries=8) for t in chunk)
            )
            vecs = [r[0] for r in sub_res]
        else:
            vecs = await _embed_nvidia_batch(client, candidate.model_id, chunk, "passage", sem, max_retries=8)
        arr = np.asarray(vecs, dtype=np.float32)
        if arr.shape != (end - start, candidate.dimension):
            raise ValueError(f"Expected shape {(end-start, candidate.dimension)}, got {arr.shape}")
        np.save(shard_path, arr)
        if (s_idx + 1) % 5 == 0 or s_idx == total_shards - 1:
            print(f"  [{candidate.model_id}] shard {s_idx+1}/{total_shards} cached ({end}/{len(texts)} rows)", flush=True)

    try:
        for s_idx in range(total_shards):
            await process_shard(s_idx)
    finally:
        await client.close()

    parts = [np.load(shards_dir / f"batch_{i:04d}.npy") for i in range(total_shards)]
    full = _l2_normalize(np.vstack(parts))
    np.save(passages_npy, full)
    meta_json.write_text(
        json.dumps({"model_id": candidate.model_id, "dataset_version": dataset_version, "rows": len(ids), "dim": candidate.dimension, "complete": True}),
        encoding="utf-8",
    )
    return full


async def _embed_nvidia_queries(candidate: CandidateModel, queries: list[str], cache_dir: Path) -> tuple[np.ndarray, list[float]]:
    q_npy = cache_dir / "dev_queries.npy"
    lat_json = cache_dir / "dev_query_latencies.json"
    if q_npy.exists() and lat_json.exists():
        arr = np.load(q_npy)
        lats = json.loads(lat_json.read_text(encoding="utf-8"))
        if arr.shape == (len(queries), candidate.dimension):
            return arr, lats

    client = AsyncOpenAI(
        api_key=os.getenv("NVIDIA_API_KEY"),
        base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/"),
        timeout=60,
        max_retries=0,
    )
    sem = asyncio.Semaphore(1)
    vecs = []
    latencies = []
    try:
        for q in queries:
            t0 = time.perf_counter()
            v = await _embed_nvidia_batch(client, candidate.model_id, [q], "query", sem)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            vecs.append(v[0])
    finally:
        await client.close()

    arr = _l2_normalize(np.asarray(vecs, dtype=np.float32))
    np.save(q_npy, arr)
    lat_json.write_text(json.dumps(latencies), encoding="utf-8")
    return arr, latencies


def _embed_pinecone_passages(
    candidate: CandidateModel,
    texts: list[str],
    ids: list[str],
    dataset_version: str,
    cache_dir: Path,
) -> np.ndarray:
    passages_npy = cache_dir / "passages.npy"
    meta_json = cache_dir / "cache_meta.json"
    if passages_npy.exists() and meta_json.exists():
        meta = json.loads(meta_json.read_text(encoding="utf-8"))
        if meta.get("dataset_version") == dataset_version and meta.get("complete") is True:
            arr = np.load(passages_npy)
            if arr.shape == (len(texts), candidate.dimension):
                print(f"  [CACHE HIT] {candidate.model_id}: loaded {arr.shape} from {passages_npy}")
                return arr

    shards_dir = cache_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    bs = candidate.batch_size
    total_batches = (len(texts) + bs - 1) // bs

    for b_idx in range(total_batches):
        shard_path = shards_dir / f"batch_{b_idx:04d}.npy"
        if shard_path.exists():
            continue
        start = b_idx * bs
        end = min(len(texts), start + bs)
        batch_texts = texts[start:end]
        for attempt in range(6):
            try:
                res = pc.inference.embed(
                    model=candidate.model_id,
                    inputs=batch_texts,
                    parameters={"input_type": "passage", "truncate": "END"},
                )
                vecs = [item.values for item in res.data]
                arr = np.asarray(vecs, dtype=np.float32)
                if arr.shape != (end - start, candidate.dimension):
                    raise ValueError(f"Expected {(end-start, candidate.dimension)}, got {arr.shape}")
                np.save(shard_path, arr)
                break
            except Exception as exc:
                status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
                if status in (400, 401, 402, 403) or attempt == 5:
                    raise
                time.sleep(1.5 * (2**attempt) + random.uniform(0.0, 0.5))
        if (b_idx + 1) % 20 == 0 or b_idx == total_batches - 1:
            print(f"  [{candidate.model_id}] batch {b_idx+1}/{total_batches} cached", flush=True)

    parts = [np.load(shards_dir / f"batch_{i:04d}.npy") for i in range(total_batches)]
    full = _l2_normalize(np.vstack(parts))
    np.save(passages_npy, full)
    meta_json.write_text(
        json.dumps({"model_id": candidate.model_id, "dataset_version": dataset_version, "rows": len(ids), "dim": candidate.dimension, "complete": True}),
        encoding="utf-8",
    )
    return full


def _embed_pinecone_queries(candidate: CandidateModel, queries: list[str], cache_dir: Path) -> tuple[np.ndarray, list[float]]:
    q_npy = cache_dir / "dev_queries.npy"
    lat_json = cache_dir / "dev_query_latencies.json"
    if q_npy.exists() and lat_json.exists():
        arr = np.load(q_npy)
        lats = json.loads(lat_json.read_text(encoding="utf-8"))
        if arr.shape == (len(queries), candidate.dimension):
            return arr, lats

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    vecs = []
    latencies = []
    for q in queries:
        t0 = time.perf_counter()
        for attempt in range(5):
            try:
                res = pc.inference.embed(
                    model=candidate.model_id,
                    inputs=[q],
                    parameters={"input_type": "query", "truncate": "END"},
                )
                latencies.append((time.perf_counter() - t0) * 1000.0)
                vecs.append(res.data[0].values)
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(1.0 * (2**attempt))

    arr = _l2_normalize(np.asarray(vecs, dtype=np.float32))
    np.save(q_npy, arr)
    lat_json.write_text(json.dumps(latencies), encoding="utf-8")
    return arr, latencies


def _is_relevant(rec: dict[str, Any], q: dict[str, Any]) -> bool:
    exp_type = (q.get("type") or "").strip().lower()
    exp_country = (q.get("country") or "").strip().lower()
    rtype = (rec.get("disaster_type") or "").lower()
    rsub = (rec.get("disaster_subtype") or "").lower()
    rcountry = (rec.get("country") or "").lower()

    type_match = exp_type in rtype or exp_type in rsub
    if not type_match:
        return False
    if exp_country:
        return exp_country == rcountry or exp_country in rcountry
    return True


def _bootstrap_ci(values: np.ndarray, n_boot: int = BOOTSTRAP_ITERATIONS, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _evaluate_model(
    candidate: CandidateModel,
    passage_vecs: np.ndarray,
    query_vecs: np.ndarray,
    latencies: list[float],
    records: list[dict[str, Any]],
    dev_queries: list[dict[str, Any]],
) -> dict[str, Any]:
    # Exact cosine dot product (both L2 normalized)
    sim_matrix = query_vecs @ passage_vecs.T  # shape (40, 16833)

    ans_rr: list[float] = []
    ans_r1: list[int] = []
    ans_r5: list[int] = []
    ans_r10: list[int] = []
    ans_top1_scores: list[float] = []
    ood_top1_scores: list[float] = []
    per_query: list[dict[str, Any]] = []

    for qi, q in enumerate(dev_queries):
        scores = sim_matrix[qi]
        top10_idx = np.argpartition(scores, -10)[-10:]
        top10_idx = top10_idx[np.argsort(scores[top10_idx])[::-1]]
        top1_score = float(scores[top10_idx[0]])

        if q["category"] == "out_of_domain":
            ood_top1_scores.append(top1_score)
            per_query.append({"id": q["id"], "category": "out_of_domain", "top1_score": round(top1_score, 4)})
            continue

        ans_top1_scores.append(top1_score)
        rank = None
        for r_idx, row_i in enumerate(top10_idx, start=1):
            if _is_relevant(records[int(row_i)], q):
                rank = r_idx
                break
        rr = 1.0 / rank if rank is not None else 0.0
        ans_rr.append(rr)
        ans_r1.append(1 if rank == 1 else 0)
        ans_r5.append(1 if rank is not None and rank <= 5 else 0)
        ans_r10.append(1 if rank is not None and rank <= 10 else 0)
        per_query.append(
            {
                "id": q["id"],
                "category": q["category"],
                "rank": rank,
                "rr": round(rr, 4),
                "top1_score": round(top1_score, 4),
                "top1_id": records[int(top10_idx[0])]["disaster_id"],
            }
        )

    rr_arr = np.asarray(ans_rr, dtype=np.float64)
    mrr = float(rr_arr.mean())
    mrr_ci = _bootstrap_ci(rr_arr)

    # Calibrate similarity reject threshold on DEV:
    # Choose threshold that keeps OOD FPs == 0 (just above max OOD top-1 score, or maximizing F1 separation)
    max_ood = max(ood_top1_scores)
    min_ans = min(ans_top1_scores)
    best_thresh = round(max_ood + 0.01, 4)
    ood_fp_at_thresh = sum(1 for s in ood_top1_scores if s >= best_thresh)
    ans_rejected_at_thresh = sum(1 for s in ans_top1_scores if s < best_thresh)

    return {
        "model_id": candidate.model_id,
        "provider": candidate.provider,
        "dimension": candidate.dimension,
        "answerable_count": len(ans_rr),
        "ood_count": len(ood_top1_scores),
        "recall@1": round(float(np.mean(ans_r1)), 4),
        "recall@5": round(float(np.mean(ans_r5)), 4),
        "recall@10": round(float(np.mean(ans_r10)), 4),
        "mrr": round(mrr, 4),
        "mrr_ci_95": [round(mrr_ci[0], 4), round(mrr_ci[1], 4)],
        "rr_vector": [round(x, 6) for x in ans_rr],
        "calibrated_threshold": best_thresh,
        "ood_max_top1_score": round(max_ood, 4),
        "ans_min_top1_score": round(min_ans, 4),
        "ood_false_positives_at_threshold": ood_fp_at_thresh,
        "ans_rejected_at_threshold": ans_rejected_at_thresh,
        "mean_query_latency_ms": round(float(np.mean(latencies)), 2),
        "p95_query_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "per_query": per_query,
    }


def _select_winner(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply pre-registered selection rule from verification/embedding_selection_rule.md."""
    eligible = [r for r in results if r["ood_false_positives_at_threshold"] <= 2]
    above_bar = [r for r in eligible if r["mrr"] >= BASELINE_MRR]
    pool = above_bar if above_bar else eligible
    pool.sort(key=lambda r: r["mrr"], reverse=True)
    best = pool[0]
    best_rr = np.asarray(best["rr_vector"], dtype=np.float64)

    tied = [best]
    pairwise_deltas = []
    for other in pool[1:]:
        diff = best_rr - np.asarray(other["rr_vector"], dtype=np.float64)
        ci_lo, ci_hi = _bootstrap_ci(diff)
        is_significant = ci_lo > 0.0
        pairwise_deltas.append(
            {
                "best_model": best["model_id"],
                "other_model": other["model_id"],
                "delta_mrr": round(float(diff.mean()), 4),
                "delta_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
                "statistically_significant": is_significant,
            }
        )
        if not is_significant:
            tied.append(other)

    # Deterministic tie-breaker per pre-registered rule:
    # 1. Reject 4096-dim if tied with <=2048-dim
    non_4096 = [r for r in tied if r["dimension"] <= 2048]
    if non_4096:
        tied = non_4096
    # 2. Prefer lower dimension (1024 over 2048), then lower mean query latency, then newer nemotron-3.5
    tied.sort(
        key=lambda r: (
            r["dimension"],
            r["mean_query_latency_ms"],
            0 if "3.5" in r["model_id"] else 1,
        )
    )
    winner = tied[0]
    return {
        "winner": winner["model_id"],
        "winner_provider": winner["provider"],
        "winner_dimension": winner["dimension"],
        "winner_mrr": winner["mrr"],
        "tied_top_models": [r["model_id"] for r in tied],
        "pairwise_deltas_vs_top_mrr": pairwise_deltas,
        "requires_index_recreate": winner["dimension"] != 2048,
    }


async def run_model_selection() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tbl = pq.read_table(SNAPSHOT_PATH)
    meta = tbl.schema.metadata or {}
    dataset_version = meta.get(b"dataset_version", b"unknown").decode("utf-8")
    records = tbl.to_pylist()
    ids = [r["disaster_id"] for r in records]
    texts = [r["search_text"] for r in records]

    dev_data = json.loads(DEV_SET_PATH.read_text(encoding="utf-8"))
    dev_queries = dev_data["retrieval_queries"]
    q_texts = [q["query"] for q in dev_queries]

    results: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for cand in CANDIDATES:
        print(f"\n=== EVALUATING CANDIDATE: {cand.model_id} ({cand.provider}, {cand.dimension}d) ===", flush=True)
        cache_dir = CACHE_ROOT / cand.slug
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            if cand.provider == "nvidia":
                p_vecs = await _embed_nvidia_passages(cand, texts, ids, dataset_version, cache_dir)
                q_vecs, lats = await _embed_nvidia_queries(cand, q_texts, cache_dir)
            else:
                p_vecs = _embed_pinecone_passages(cand, texts, ids, dataset_version, cache_dir)
                q_vecs, lats = _embed_pinecone_queries(cand, q_texts, cache_dir)
            res = _evaluate_model(cand, p_vecs, q_vecs, lats, records, dev_queries)
            print(
                f"  -> R@1={res['recall@1']:.4f}, R@5={res['recall@5']:.4f}, R@10={res['recall@10']:.4f}, "
                f"MRR={res['mrr']:.4f} {res['mrr_ci_95']}, thresh={res['calibrated_threshold']}, "
                f"OOD_FP={res['ood_false_positives_at_threshold']}/{res['ood_count']}, lat={res['mean_query_latency_ms']}ms",
                flush=True,
            )
            results.append(res)
        except Exception as exc:
            print(f"  [STOP/ERROR] {cand.model_id} failed: {type(exc).__name__}: {exc}", flush=True)
            skipped.append({"model_id": cand.model_id, "error": f"{type(exc).__name__}: {exc}"})

    selection = _select_winner(results) if results else {"winner": None}
    payload = {
        "dataset_version": dataset_version,
        "snapshot_count": len(records),
        "dev_answerable_queries": sum(1 for q in dev_queries if q["category"] != "out_of_domain"),
        "dev_ood_queries": sum(1 for q in dev_queries if q["category"] == "out_of_domain"),
        "results": [{k: v for k, v in r.items() if k != "rr_vector"} for r in results],
        "skipped": skipped,
        "selection": selection,
    }
    (OUT_DIR / "model_selection_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        "# Task 2P Step 8 — Embedding Model Selection Report (DEV Set Exact Cosine)",
        "",
        f"- **Snapshot Version:** `{dataset_version}` ({len(records):,} records)",
        f"- **DEV Suite:** {payload['dev_answerable_queries']} answerable queries + {payload['dev_ood_queries']} OOD queries",
        f"- **Winner (per pre-registered rule):** `{selection.get('winner')}` ({selection.get('winner_provider')}, {selection.get('winner_dimension')}-dim)",
        f"- **Requires Index Recreation (`dim != 2048`):** {selection.get('requires_index_recreate')}",
        "",
        "| Model | Provider | Dim | R@1 | R@5 | R@10 | MRR (95% CI) | Calib. Threshold | OOD FP | Mean Query Latency |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for r in payload["results"]:
        ci = r["mrr_ci_95"]
        md.append(
            f"| `{r['model_id']}` | {r['provider']} | {r['dimension']} | "
            f"{r['recall@1']:.4f} | {r['recall@5']:.4f} | {r['recall@10']:.4f} | "
            f"**{r['mrr']:.4f}** [{ci[0]:.4f}, {ci[1]:.4f}] | "
            f"{r['calibrated_threshold']:.4f} | {r['ood_false_positives_at_threshold']}/{r['ood_count']} | {r['mean_query_latency_ms']:.1f} ms |"
        )
    if skipped:
        md.extend(["", "## Skipped / Quota Errors", ""])
        for s in skipped:
            md.append(f"- `{s['model_id']}`: `{s['error']}`")
    md.extend(["", "## Pairwise Paired Bootstrap Deltas vs. Top-MRR Model", ""])
    for pd_item in selection.get("pairwise_deltas_vs_top_mrr", []):
        md.append(
            f"- `{pd_item['best_model']}` vs `{pd_item['other_model']}`: "
            f"delta_MRR={pd_item['delta_mrr']:+.4f}, 95% CI={pd_item['delta_ci_95']}, "
            f"significant={pd_item['statistically_significant']}"
        )

    (OUT_DIR / "model_selection_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n[MODEL SELECTION COMPLETE] Winner: {selection.get('winner')}")
    return payload


if __name__ == "__main__":
    asyncio.run(run_model_selection())

