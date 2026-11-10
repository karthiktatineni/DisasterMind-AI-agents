"""Invariant D4: Startup integrity validation against canonical snapshot and ingestion ledger."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from backend.app.config import Settings, get_settings

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SNAPSHOT_PATH = PROJECT_ROOT / "data" / "disastermind_snapshot.parquet"
DEFAULT_LEDGER_PATH = PROJECT_ROOT / "data" / "ingestion_ledger.json"


class StartupIntegrityError(RuntimeError):
    """Raised when snapshot dataset_version, embedding model, or namespace mismatches config (D4)."""


def read_snapshot_version(snapshot_path: Path | None = None) -> str:
    snap_p = snapshot_path or DEFAULT_SNAPSHOT_PATH
    if not snap_p.exists():
        raise StartupIntegrityError(f"Canonical snapshot missing: {snap_p}")
    schema = pq.read_schema(snap_p)
    meta = schema.metadata or {}
    raw = meta.get(b"dataset_version")
    if not raw:
        raise StartupIntegrityError(f"Snapshot missing dataset_version metadata: {snap_p}")
    return raw.decode("utf-8")


def verify_runtime_alignment(
    settings: Settings | None = None,
    *,
    snapshot_path: Path | None = None,
    ledger_path: Path | None = None,
    require_ingested_ledger: bool = False,
) -> dict[str, str]:
    """Verify snapshot dataset_version, configured embedding model, dimension, and namespace match ledger."""
    cfg = settings or get_settings()
    snap_p = snapshot_path or DEFAULT_SNAPSHOT_PATH
    led_p = ledger_path or DEFAULT_LEDGER_PATH

    if not led_p.exists():
        if require_ingested_ledger:
            raise StartupIntegrityError(f"Ingestion ledger missing: {led_p}")
        return {"status": "uninitialized", "reason": "No ingestion ledger yet."}

    snap_version = read_snapshot_version(snap_p)
    ledger = json.loads(led_p.read_text(encoding="utf-8"))

    ledger_version = ledger.get("dataset_version")
    ledger_model = ledger.get("embedding_model")
    ledger_namespace = ledger.get("pinecone_namespace")
    ledger_dim = ledger.get("embedding_dimension")

    if ledger_version != snap_version:
        raise StartupIntegrityError(
            f"dataset_version mismatch: snapshot='{snap_version}' != ledger='{ledger_version}'"
        )
    if ledger_model != cfg.nvidia_embedding_model:
        raise StartupIntegrityError(
            f"embedding_model mismatch: config='{cfg.nvidia_embedding_model}' != ledger='{ledger_model}'"
        )
    if ledger_namespace != cfg.pinecone_namespace:
        raise StartupIntegrityError(
            f"pinecone_namespace mismatch: config='{cfg.pinecone_namespace}' != ledger='{ledger_namespace}'"
        )
    if ledger_dim is not None and int(ledger_dim) != int(cfg.nvidia_embedding_dimension):
        raise StartupIntegrityError(
            f"embedding_dimension mismatch: config={cfg.nvidia_embedding_dimension} != ledger={ledger_dim}"
        )

    return {
        "status": "ok",
        "dataset_version": snap_version,
        "embedding_model": str(ledger_model),
        "pinecone_namespace": str(ledger_namespace),
        "embedding_dimension": str(ledger_dim),
    }

