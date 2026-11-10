"""Task 2P Steps 9-11: Resumable Idempotent Pinecone Ingestion & Full D1-D5 Verification."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import orjson
import pyarrow.parquet as pq
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env", override=False)

from backend.app.services.embedding_service import (  # noqa: E402
    INPUT_TYPE_PASSAGE,
    SAFE_CLIENT_TOKEN_LIMIT,
    EmbeddingService,
    _truncate_to_token_limit,
)
from backend.app.config import Settings  # noqa: E402

SNAPSHOT_PATH = PROJECT_ROOT / "data" / "disastermind_snapshot.parquet"
LEDGER_PATH = PROJECT_ROOT / "data" / "ingestion_ledger.json"
CACHE_ROOT = PROJECT_ROOT / "eval" / ".cache"
SELECTION_PATH = PROJECT_ROOT / "verification" / "task_2p" / "model_selection_results.json"
OUT_DIR = PROJECT_ROOT / "verification" / "task_2p"
MAX_PAYLOAD_BYTES = 1_800_000  # strictly below Pinecone 2 MB REST limit


def _vector_sha256(vec: list[float] | np.ndarray) -> str:
    arr = np.asarray(vec, dtype=np.float32)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def build_pinecone_metadata(rec: dict[str, Any]) -> dict[str, Any]:
    """Build Pinecone-compliant metadata dict (allowed types only, nulls omitted, no 'unknown')."""
    raw_meta = {
        "disaster_id": str(rec["disaster_id"]),
        "dataset_version": str(rec["dataset_version"]),
        "disaster_type": str(rec["disaster_type"]),
        "disaster_subtype": rec.get("disaster_subtype"),
        "category_tag": str(rec["category_tag"]),
        "event_name": str(rec["event_name"]),
        "display_label": str(rec["display_label"]),
        "country": str(rec["country"]),
        "region": rec.get("region"),
        "location": rec.get("location"),
        "start_year": int(rec["start_year"]),
        "total_deaths": rec.get("total_deaths"),
        "total_affected": rec.get("total_affected"),
        "total_damage": rec.get("total_damage_usd_thousands"),
        "search_text": str(rec["search_text"]),
    }
    clean: dict[str, Any] = {}
    for k, v in raw_meta.items():
        if v is None:
            continue
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            continue
        if isinstance(v, str):
            if "unknown" in v.lower():
                raise ValueError(f"Invariant D2 violated: 'unknown' in metadata['{k}'] for {rec['disaster_id']}: {v!r}")
            clean[k] = v
        elif isinstance(v, (bool, int, float)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean


def _resolve_target_model(model_override: str | None = None) -> tuple[str, str, int]:
    if SELECTION_PATH.exists() and not model_override:
        sel = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))["selection"]
        if sel.get("winner"):
            return sel["winner"], sel["winner_provider"], int(sel["winner_dimension"])
    model_id = model_override or os.getenv("NVIDIA_EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b")
    if model_id in ("llama-text-embed-v2", "multilingual-e5-large"):
        return model_id, "pinecone", 1024
    if "8b" in model_id:
        return model_id, "nvidia", 4096
    return model_id, "nvidia", 2048


def _load_cached_vectors(model_id: str, dataset_version: str, expected_rows: int, expected_dim: int) -> np.ndarray | None:
    slug = model_id.replace("/", "__")
    cdir = CACHE_ROOT / slug
    meta_p = cdir / "cache_meta.json"
    npy_p = cdir / "passages.npy"
    if meta_p.exists() and npy_p.exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if meta.get("complete") and meta.get("dataset_version") == dataset_version and meta.get("model_id") == model_id:
            arr = np.load(npy_p)
            if arr.shape == (expected_rows, expected_dim):
                return arr
    return None


async def _embed_missing_nvidia(model_id: str, dim: int, texts: list[str]) -> list[list[float]]:
    cfg = Settings(
        nvidia_api_key=os.getenv("NVIDIA_API_KEY", ""),
        nvidia_embedding_model=model_id,
        nvidia_embedding_dimension=dim,
        nvidia_base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
    )
    svc = EmbeddingService(settings=cfg)
    out: list[list[float]] = []
    bs = 50 if "3-embed" in model_id else 1
    try:
        for i in range(0, len(texts), bs):
            chunk = [_truncate_to_token_limit(t, SAFE_CLIENT_TOKEN_LIMIT) for t in texts[i : i + bs]]
            vecs = await svc.embed_texts(chunk, input_type=INPUT_TYPE_PASSAGE)
            out.extend(vecs)
    finally:
        await svc._client.close()
    return out


def _embed_missing_pinecone(pc: Pinecone, model_id: str, texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), 96):
        chunk = texts[i : i + 96]
        for attempt in range(6):
            try:
                res = pc.inference.embed(
                    model=model_id,
                    inputs=chunk,
                    parameters={"input_type": INPUT_TYPE_PASSAGE, "truncate": "END"},
                )
                out.extend([item.values for item in res.data])
                break
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(1.5 * (2**attempt) + random.uniform(0.0, 0.5))
    return out


def _ensure_index(pc: Pinecone, index_name: str, target_dim: int) -> Any:
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name in existing:
        desc = pc.describe_index(index_name)
        if int(desc.dimension) != int(target_dim):
            stats = pc.Index(index_name).describe_index_stats()
            if int(stats.get("total_vector_count", 0)) > 0:
                raise RuntimeError(
                    f"Index '{index_name}' has dim={desc.dimension} with {stats['total_vector_count']} vectors; "
                    f"refusing automatic delete when non-empty (target_dim={target_dim})."
                )
            print(f"[INDEX] Recreating empty index '{index_name}' from dim={desc.dimension} to dim={target_dim}...")
            pc.delete_index(index_name)
            for _ in range(30):
                if index_name not in [x.name for x in pc.list_indexes()]:
                    break
                time.sleep(2)
            pc.create_index(
                name=index_name,
                dimension=target_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            for _ in range(60):
                d = pc.describe_index(index_name)
                st = d.status.get("state") if isinstance(d.status, dict) else d.status.state
                if str(st).lower() == "ready":
                    break
                time.sleep(2)
    else:
        pc.create_index(
            name=index_name,
            dimension=target_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        for _ in range(60):
            d = pc.describe_index(index_name)
            st = d.status.get("state") if isinstance(d.status, dict) else d.status.state
            if str(st).lower() == "ready":
                break
            time.sleep(2)
    return pc.Index(index_name)


def run_ingestion(
    *,
    model_override: str | None = None,
    namespace_override: str | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    tbl = pq.read_table(SNAPSHOT_PATH)
    dataset_version = (tbl.schema.metadata or {}).get(b"dataset_version", b"").decode("utf-8")
    records = tbl.to_pylist()
    model_id, provider, dim = _resolve_target_model(model_override)
    slug = model_id.replace("/", "__")
    namespace = namespace_override or f"v2-{dataset_version}-{slug}"
    index_name = os.getenv("PINECONE_INDEX_NAME", "disastermind").strip()

    ledger: dict[str, Any] = {}
    if LEDGER_PATH.exists():
        raw_ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
        if (
            raw_ledger.get("dataset_version") == dataset_version
            and raw_ledger.get("embedding_model") == model_id
            and raw_ledger.get("pinecone_namespace") == namespace
        ):
            ledger = raw_ledger
        else:
            ledger = {}
    if not ledger:
        ledger = {
            "dataset_version": dataset_version,
            "embedding_model": model_id,
            "provider": provider,
            "embedding_dimension": dim,
            "pinecone_index_name": index_name,
            "pinecone_namespace": namespace,
            "records": {},
        }

    rec_ledger: dict[str, dict[str, Any]] = ledger["records"]

    # Determine pending records (not yet embedded & upserted with matching search_text_sha256)
    pending_indices: list[int] = []
    for idx, rec in enumerate(records):
        did = rec["disaster_id"]
        entry = rec_ledger.get(did)
        if (
            entry is not None
            and entry.get("search_text_sha256") == rec["search_text_sha256"]
            and entry.get("upserted") is True
            and entry.get("vector_sha256")
        ):
            continue
        pending_indices.append(idx)

    print(
        f"[INGEST] dataset_version={dataset_version}, model={model_id} ({provider}, {dim}d), "
        f"namespace={namespace}, total={len(records)}, pending={len(pending_indices)}",
        flush=True,
    )

    if not pending_indices:
        print("[INGEST IDEMPOTENT] 0 pending records: 0 embedding calls, 0 upserts.", flush=True)
        return {
            "dataset_version": dataset_version,
            "model_id": model_id,
            "provider": provider,
            "dimension": dim,
            "namespace": namespace,
            "embedded_count": 0,
            "upserted_count": 0,
            "total_in_ledger": len(rec_ledger),
        }

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = _ensure_index(pc, index_name, dim)

    cached_arr = _load_cached_vectors(model_id, dataset_version, len(records), dim) if use_cache else None
    embedded_calls = 0
    if cached_arr is not None:
        print(f"[INGEST] Using validated local passage cache for {model_id} ({cached_arr.shape})", flush=True)
        vectors_map = {idx: cached_arr[idx].tolist() for idx in pending_indices}
    else:
        pending_texts = [records[idx]["search_text"] for idx in pending_indices]
        if provider == "nvidia":
            vecs = asyncio.run(_embed_missing_nvidia(model_id, dim, pending_texts))
        else:
            vecs = _embed_missing_pinecone(pc, model_id, pending_texts)
        embedded_calls = len(pending_texts)
        vectors_map = {}
        for idx, vec in zip(pending_indices, vecs, strict=True):
            arr = np.asarray(vec, dtype=np.float32)
            norm = float(np.linalg.norm(arr))
            if norm > 0:
                arr = arr / norm
            vectors_map[idx] = arr.tolist()

    # Batch upsert respecting measured 2MB payload limit
    batch_size = 50 if dim <= 2048 else 25
    upserted_total = 0

    def _send_batch(batch_items: list[dict[str, Any]]) -> int:
        payload_bytes = len(orjson.dumps({"vectors": batch_items, "namespace": namespace}))
        if payload_bytes > MAX_PAYLOAD_BYTES and len(batch_items) > 1:
            mid = len(batch_items) // 2
            return _send_batch(batch_items[:mid]) + _send_batch(batch_items[mid:])
        for attempt in range(5):
            try:
                resp = index.upsert(vectors=batch_items, namespace=namespace)
                return int(getattr(resp, "upserted_count", len(batch_items)))
            except Exception as exc:
                if attempt == 4:
                    raise
                time.sleep(1.5 * (2**attempt) + random.uniform(0.0, 0.5))
        return 0

    for start in range(0, len(pending_indices), batch_size):
        chunk_indices = pending_indices[start : start + batch_size]
        upsert_vectors = []
        for idx in chunk_indices:
            rec = records[idx]
            meta = build_pinecone_metadata(rec)
            upsert_vectors.append(
                {
                    "id": rec["disaster_id"],
                    "values": vectors_map[idx],
                    "metadata": meta,
                }
            )
        count = _send_batch(upsert_vectors)
        upserted_total += count

        for idx in chunk_indices:
            rec = records[idx]
            rec_ledger[rec["disaster_id"]] = {
                "search_text_sha256": rec["search_text_sha256"],
                "vector_sha256": _vector_sha256(vectors_map[idx]),
                "upserted": True,
            }

        # Atomic checkpoint save every 25 batches or end
        if ((start // batch_size) + 1) % 25 == 0 or start + batch_size >= len(pending_indices):
            tmp = LEDGER_PATH.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(ledger), encoding="utf-8")
            tmp.replace(LEDGER_PATH)
            print(f"  [UPSERT] {upserted_total}/{len(pending_indices)} upserted", flush=True)

    return {
        "dataset_version": dataset_version,
        "model_id": model_id,
        "provider": provider,
        "dimension": dim,
        "namespace": namespace,
        "embedded_count": embedded_calls if cached_arr is None else 0,
        "used_cached_embeddings": cached_arr is not None,
        "upserted_count": upserted_total,
        "total_in_ledger": len(rec_ledger),
    }


def run_full_d1_d5_verification() -> dict[str, Any]:
    """Step 11: Full live verification against Pinecone + D3 second-run network capture."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tbl = pq.read_table(SNAPSHOT_PATH)
    records = tbl.to_pylist()
    snap_by_id = {r["disaster_id"]: r for r in records}
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    namespace = ledger["pinecone_namespace"]
    index_name = ledger["pinecone_index_name"]
    expected_dim = int(ledger["embedding_dimension"])
    model_id = ledger["embedding_model"]
    dataset_version = ledger["dataset_version"]

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(index_name)

    # 1. Poll for eventual consistency of count
    ns_count = 0
    for _ in range(30):
        stats = index.describe_index_stats()
        ns_info = stats.get("namespaces", {}).get(namespace, {})
        ns_count = int(ns_info.get("vector_count", 0))
        if ns_count == len(records):
            break
        time.sleep(2)

    assert ns_count == len(records), f"D1 count mismatch: pinecone={ns_count} != snapshot={len(records)}"
    assert len(ledger["records"]) == len(records), f"Ledger count mismatch: {len(ledger['records'])} != {len(records)}"

    # 2. List all IDs in namespace
    pinecone_ids: list[str] = []
    for page in index.list(namespace=namespace):
        if hasattr(page, "vectors") and page.vectors is not None:
            pinecone_ids.extend(
                [item.id if hasattr(item, "id") else str(item) for item in page.vectors]
            )
        elif isinstance(page, list):
            pinecone_ids.extend([str(x) for x in page])
        else:
            pinecone_ids.append(str(page))
    assert set(pinecone_ids) == set(snap_by_id.keys()), (
        f"D1 ID set mismatch: only_pc={len(set(pinecone_ids)-set(snap_by_id))}, "
        f"only_snap={len(set(snap_by_id)-set(pinecone_ids))}"
    )

    # 3. Fetch-and-compare ALL 16,833 metadata records and vector norms/hashes
    cached_arr = _load_cached_vectors(model_id, dataset_version, len(records), expected_dim)
    snap_idx_by_id = {r["disaster_id"]: i for i, r in enumerate(records)}

    all_ids = sorted(snap_by_id.keys())
    metadata_mismatches = 0
    vector_mismatches = 0
    unknown_hits = 0
    checked = 0

    for i in range(0, len(all_ids), 100):
        batch_ids = all_ids[i : i + 100]
        fetched = index.fetch(ids=batch_ids, namespace=namespace)
        vectors_dict = fetched.vectors if hasattr(fetched, "vectors") else fetched.get("vectors", {})
        for did in batch_ids:
            vobj = vectors_dict.get(did)
            if vobj is None:
                metadata_mismatches += 1
                continue
            checked += 1
            f_meta = vobj.metadata if hasattr(vobj, "metadata") else vobj.get("metadata", {})
            f_vals = vobj.values if hasattr(vobj, "values") else vobj.get("values", [])
            exp_meta = build_pinecone_metadata(snap_by_id[did])

            if f_meta != exp_meta:
                # Allow int/float equivalence for numeric fields returned as floats by JSON
                diff_real = False
                if set(f_meta.keys()) != set(exp_meta.keys()):
                    diff_real = True
                else:
                    for k, ev in exp_meta.items():
                        fv = f_meta.get(k)
                        if isinstance(ev, (int, float)) and isinstance(fv, (int, float)):
                            if abs(float(ev) - float(fv)) > 1e-4:
                                diff_real = True
                        elif fv != ev:
                            diff_real = True
                if diff_real:
                    metadata_mismatches += 1

            for mv in f_meta.values():
                if isinstance(mv, str) and "unknown" in mv.lower():
                    unknown_hits += 1

            arr_f = np.asarray(f_vals, dtype=np.float32)
            if len(arr_f) != expected_dim or not np.all(np.isfinite(arr_f)):
                vector_mismatches += 1
            elif cached_arr is not None:
                ref_arr = cached_arr[snap_idx_by_id[did]]
                cos = float(np.dot(arr_f / np.linalg.norm(arr_f), ref_arr))
                if cos < 0.9999:
                    vector_mismatches += 1

        if (i // 100 + 1) % 25 == 0 or i + 100 >= len(all_ids):
            print(f"  [D1 FETCH CHECK] {checked}/{len(all_ids)} verified", flush=True)

    # 4. Positive-Control httpx Transport Capture (intercepts wire HTTPTransport / AsyncHTTPTransport
    # including Pinecone SDK 10.3.0 HTTPClient.post fast path that calls _transport.handle_request directly)
    orig_sync_transport = httpx.HTTPTransport.handle_request
    orig_async_transport = httpx.AsyncHTTPTransport.handle_async_request

    class _WireSpy:
        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []

        def __enter__(self) -> list[dict[str, Any]]:
            bucket = self.requests

            def spy_sync(transport_self, request: httpx.Request, *args, **kwargs):
                resp = orig_sync_transport(transport_self, request, *args, **kwargs)
                bucket.append({"transport": "sync", "method": request.method, "url": str(request.url), "status_code": resp.status_code})
                return resp

            async def spy_async(transport_self, request: httpx.Request, *args, **kwargs):
                resp = await orig_async_transport(transport_self, request, *args, **kwargs)
                bucket.append({"transport": "async", "method": request.method, "url": str(request.url), "status_code": resp.status_code})
                return resp

            httpx.HTTPTransport.handle_request = spy_sync  # type: ignore[method-assign]
            httpx.AsyncHTTPTransport.handle_async_request = spy_async  # type: ignore[method-assign]
            return self.requests

        def __exit__(self, *exc_info: Any) -> None:
            httpx.HTTPTransport.handle_request = orig_sync_transport  # type: ignore[method-assign]
            httpx.AsyncHTTPTransport.handle_async_request = orig_async_transport  # type: ignore[method-assign]

    ctrl_ns = "__d3_positive_control_tmp"
    with _WireSpy() as ctrl_requests:
        ctrl_pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        ctrl_idx = ctrl_pc.Index(index_name)
        sample_rec = records[0]
        if ledger.get("provider") == "pinecone":
            emb_res = ctrl_pc.inference.embed(
                model=model_id,
                inputs=[sample_rec["search_text"]],
                parameters={"input_type": INPUT_TYPE_PASSAGE, "truncate": "END"},
            )
            ctrl_vec = list(emb_res.data[0].values)
        else:
            ctrl_vec = cached_arr[0].tolist() if cached_arr is not None else [0.01] * expected_dim
        ctrl_idx.upsert(
            vectors=[{"id": f"ctrl-{sample_rec['disaster_id']}", "values": ctrl_vec, "metadata": build_pinecone_metadata(sample_rec)}],
            namespace=ctrl_ns,
        )
        ctrl_idx.delete(delete_all=True, namespace=ctrl_ns)

    ctrl_embed_calls = [r for r in ctrl_requests if "/embeddings" in r["url"] or "/embed" in r["url"]]
    ctrl_upsert_calls = [r for r in ctrl_requests if "/upsert" in r["url"]]
    (OUT_DIR / "first_ingest_positive_control_capture.json").write_text(
        json.dumps(
            {
                "note": "Positive-control wire capture using httpx HTTPTransport/AsyncHTTPTransport spy on a temporary namespace (cleaned via delete_all). Original 16,833-record ingest (task-680) logged stdout progress without enabling this HTTP transport capture hook.",
                "control_namespace": ctrl_ns,
                "total_http_requests": len(ctrl_requests),
                "embedding_calls": len(ctrl_embed_calls),
                "upsert_calls": len(ctrl_upsert_calls),
                "requests": ctrl_requests,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # 5. D3 Second-Run Idempotency with Recorded Wire-Level httpx Network Capture
    with _WireSpy() as captured_requests:
        second_run = run_ingestion(model_override=model_id, namespace_override=namespace)

    embedding_http_calls = [r for r in captured_requests if "/embeddings" in r["url"] or "/embed" in r["url"]]
    upsert_http_calls = [r for r in captured_requests if "/upsert" in r["url"]]

    # 6. Explicit D2, D4, D5 live verification checks
    from backend.app.services.data_pipeline import compute_content_hash

    empty_search_text_count = sum(1 for r in records if not str(r.get("search_text", "")).strip())
    dup_id_count = len(records) - len({r["disaster_id"] for r in records})
    dup_hash_count = len(records) - len({compute_content_hash(r) for r in records})

    from backend.app.config import Settings
    from backend.app.services.historical_stats import HistoricalStatsService
    from backend.app.services.pinecone_client import PineconeVectorClient
    from backend.app.services.snapshot_guard import StartupIntegrityError, verify_runtime_alignment

    live_cfg = Settings(
        nvidia_api_key=os.getenv("NVIDIA_API_KEY", ""),
        nvidia_embedding_model=model_id,
        nvidia_embedding_dimension=expected_dim,
        pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
        pinecone_index_name=index_name,
        pinecone_namespace=namespace,
    )
    d4_ok = verify_runtime_alignment(live_cfg, snapshot_path=SNAPSHOT_PATH, ledger_path=LEDGER_PATH)
    d4_refused: list[str] = []
    for label, cfg in [
        ("embedding_model", Settings(nvidia_api_key="x", nvidia_embedding_model="wrong-model", nvidia_embedding_dimension=expected_dim, pinecone_namespace=namespace)),
        ("pinecone_namespace", Settings(nvidia_api_key="x", nvidia_embedding_model=model_id, nvidia_embedding_dimension=expected_dim, pinecone_namespace="wrong-ns")),
    ]:
        try:
            verify_runtime_alignment(cfg, snapshot_path=SNAPSHOT_PATH, ledger_path=LEDGER_PATH)
        except StartupIntegrityError as exc:
            d4_refused.append(f"{label}: {exc}")
    bad_ver_ledger = PROJECT_ROOT / "data" / ".tmp_bad_ver_ledger.json"
    bad_ver_ledger.write_text(json.dumps({**{k: v for k, v in ledger.items() if k != "records"}, "dataset_version": "0000000000000000"}), encoding="utf-8")
    try:
        try:
            verify_runtime_alignment(live_cfg, snapshot_path=SNAPSHOT_PATH, ledger_path=bad_ver_ledger)
        except StartupIntegrityError as exc:
            d4_refused.append(f"dataset_version: {exc}")
    finally:
        bad_ver_ledger.unlink(missing_ok=True)

    # Verify D5 empty-metadata skipping + sample live metadata completeness + historical_stats n<5 guardrail
    class _FakeEmptyMetaPC:
        def Index(self, _name):
            class _Idx:
                def query(self, **kw):
                    return {
                        "matches": [
                            {"id": "no-meta", "score": 0.99, "metadata": {}},
                            {"id": "with-meta", "score": 0.85, "metadata": {"event_name": "Valid", "disaster_type": "Storm", "country": "India", "start_year": 2020}},
                        ]
                    }
            return _Idx()

    d5_client = PineconeVectorClient(settings=live_cfg, pc=_FakeEmptyMetaPC())
    d5_rows = asyncio.run(d5_client.query([0.0] * expected_dim, top_k=2))
    d5_skipped_empty = len(d5_rows) == 1 and d5_rows[0]["id"] == "with-meta"
    d5_hist_n4 = HistoricalStatsService(records=records[:4]).query_stats()["status"] == "no_data"

    d1_pass = (
        ns_count == len(records) == len(ledger["records"])
        and checked == len(records)
        and metadata_mismatches == 0
        and vector_mismatches == 0
    )
    d2_pass = (
        empty_search_text_count == 0
        and unknown_hits == 0
        and dup_id_count == 0
        and dup_hash_count == 0
    )
    d3_pass = (
        second_run.get("embedded_count") == 0
        and second_run.get("upserted_count") == 0
        and len(captured_requests) == 0
        and len(embedding_http_calls) == 0
        and len(upsert_http_calls) == 0
        and len(ctrl_upsert_calls) >= 1
    )
    d4_pass = d4_ok.get("status") == "ok" and len(d4_refused) == 3
    d5_pass = d5_skipped_empty and d5_hist_n4 and metadata_mismatches == 0

    report = {
        "dataset_version": dataset_version,
        "model_id": model_id,
        "dimension": expected_dim,
        "namespace": namespace,
        "snapshot_count": len(records),
        "pinecone_namespace_count": ns_count,
        "ledger_count": len(ledger["records"]),
        "id_sets_equal": True,
        "fetched_checked": checked,
        "metadata_mismatches": metadata_mismatches,
        "vector_mismatches": vector_mismatches,
        "unknown_literal_hits": unknown_hits,
        "d1": {
            "pass": d1_pass,
            "snapshot_count": len(records),
            "pinecone_namespace_count": ns_count,
            "ledger_count": len(ledger["records"]),
            "id_sets_equal": True,
            "fetched_checked": checked,
            "metadata_mismatches": metadata_mismatches,
            "vector_mismatches": vector_mismatches,
        },
        "d2": {
            "pass": d2_pass,
            "empty_search_text_count": empty_search_text_count,
            "unknown_literal_hits": unknown_hits,
            "duplicate_id_count": dup_id_count,
            "duplicate_content_hash_count": dup_hash_count,
        },
        "d3": {
            "pass": d3_pass,
            "positive_control_http_requests": len(ctrl_requests),
            "positive_control_upsert_calls": len(ctrl_upsert_calls),
            "second_run_result": second_run,
            "second_run_total_http_calls": len(captured_requests),
            "second_run_embedding_http_calls": len(embedding_http_calls),
            "second_run_upsert_http_calls": len(upsert_http_calls),
        },
        "d4": {
            "pass": d4_pass,
            "live_alignment": d4_ok,
            "refused_mismatches": d4_refused,
        },
        "d5": {
            "pass": d5_pass,
            "empty_metadata_match_skipped": d5_skipped_empty,
            "historical_stats_n_below_5_no_data": d5_hist_n4,
            "live_metadata_parity_checked": checked,
        },
        "all_d1_d5_pass": d1_pass and d2_pass and d3_pass and d4_pass and d5_pass,
    }
    (OUT_DIR / "d1_d5_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT_DIR / "second_run_network_capture.json").write_text(
        json.dumps(
            {
                "total_http_requests": len(captured_requests),
                "embedding_calls": len(embedding_http_calls),
                "upsert_calls": len(upsert_http_calls),
                "requests": captured_requests,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[D1-D5 VERIFY] all_pass={report['all_d1_d5_pass']} d1={d1_pass} d2={d2_pass} d3={d3_pass} d4={d4_pass} d5={d5_pass} checked={checked} ctrl_http={len(ctrl_requests)} 2nd_run_http={len(captured_requests)}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["ingest", "verify", "all"], default="all", nargs="?")
    parser.add_argument("--model", default=None)
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    if args.action in ("ingest", "all"):
        run_ingestion(model_override=args.model, namespace_override=args.namespace, use_cache=not args.no_cache)
    if args.action in ("verify", "all"):
        run_full_d1_d5_verification()
    return 0


if __name__ == "__main__":
    sys.exit(main())

