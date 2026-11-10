"""Invariant D4: Startup integrity validation against committed manifest, snapshot, ledger, and live Pinecone."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from pinecone import Pinecone

from backend.app.config import Settings, get_settings

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "config" / "index_manifest.json"
DEFAULT_SNAPSHOT_PATH = PROJECT_ROOT / "data" / "disastermind_snapshot.parquet"
DEFAULT_LEDGER_PATH = PROJECT_ROOT / "data" / "ingestion_ledger.json"


class StartupIntegrityError(RuntimeError):
    """Raised when manifest, snapshot, ledger, config, or live Pinecone index mismatches (D4)."""


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


def _verify_live_pinecone(
    cfg: Settings,
    manifest: dict[str, Any],
    *,
    pinecone_stats: dict[str, Any] | None = None,
    index_description: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify live Pinecone index dimension, namespace existence, and remote vector count match manifest."""
    if index_description is None or pinecone_stats is None:
        if not cfg.pinecone_api_key:
            raise StartupIntegrityError("PINECONE_API_KEY is required to verify live index manifest alignment.")
        try:
            pc = Pinecone(api_key=cfg.pinecone_api_key, timeout=15.0)
            if index_description is None:
                desc = pc.describe_index(cfg.pinecone_index_name)
                index_description = {
                    "name": getattr(desc, "name", cfg.pinecone_index_name),
                    "dimension": int(getattr(desc, "dimension", 0)),
                    "metric": str(getattr(desc, "metric", "")),
                }
            if pinecone_stats is None:
                raw_stats = pc.Index(cfg.pinecone_index_name).describe_index_stats()
                pinecone_stats = raw_stats.to_dict() if hasattr(raw_stats, "to_dict") else dict(raw_stats)
        except Exception as exc:
            raise StartupIntegrityError(f"Live Pinecone verification failed: {exc}") from exc

    exp_dim = int(manifest["embedding_dimension"])
    remote_dim = index_description.get("dimension")
    if remote_dim is None or int(remote_dim) != exp_dim:
        raise StartupIntegrityError(
            f"Pinecone index dimension mismatch: remote={remote_dim} != manifest={exp_dim}"
        )

    exp_ns = str(manifest["pinecone_namespace"])
    namespaces = (pinecone_stats or {}).get("namespaces") or {}
    if exp_ns not in namespaces:
        raise StartupIntegrityError(
            f"Pinecone namespace missing: '{exp_ns}' not found in live index '{cfg.pinecone_index_name}'"
        )

    remote_count = int((namespaces.get(exp_ns) or {}).get("vector_count", 0))
    exp_count = int(manifest["record_count"])
    if remote_count != exp_count:
        raise StartupIntegrityError(
            f"Pinecone namespace vector count mismatch: remote={remote_count} != manifest={exp_count}"
        )

    return {"remote_dimension": int(remote_dim), "remote_vector_count": remote_count}


def verify_runtime_alignment(
    settings: Settings | None = None,
    *,
    manifest_path: Path | None = None,
    snapshot_path: Path | None = None,
    ledger_path: Path | None = None,
    require_ingested_ledger: bool = False,
    verify_remote: bool | None = None,
    pinecone_stats: dict[str, Any] | None = None,
    index_description: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify committed manifest, optional local snapshot/ledger, config, and live Pinecone match.

    In production (APP_ENV=production), a missing manifest or inability to verify live Pinecone
    always refuses startup; deleting the local ledger cannot bypass this check.
    """
    cfg = settings or get_settings()
    man_p = manifest_path or DEFAULT_MANIFEST_PATH
    snap_p = snapshot_path or DEFAULT_SNAPSHOT_PATH
    led_p = ledger_path or DEFAULT_LEDGER_PATH

    if not man_p.exists():
        if cfg.production or require_ingested_ledger:
            raise StartupIntegrityError(f"Committed index manifest missing: {man_p}")
        if not led_p.exists():
            return {"status": "uninitialized", "reason": "No manifest or ingestion ledger present (non-production)."}
        manifest = None
    else:
        manifest = json.loads(man_p.read_text(encoding="utf-8"))

    if manifest is not None:
        m_ver = str(manifest["dataset_version"])
        m_provider = str(manifest["embedding_provider"])
        m_model = str(manifest["embedding_model"])
        m_dim = int(manifest["embedding_dimension"])
        m_idx = str(manifest["pinecone_index_name"])
        m_ns = str(manifest["pinecone_namespace"])

        if cfg.embedding_provider.lower() != m_provider.lower():
            raise StartupIntegrityError(
                f"embedding_provider mismatch: config='{cfg.embedding_provider}' != manifest='{m_provider}'"
            )
        if cfg.embedding_model != m_model:
            raise StartupIntegrityError(
                f"embedding_model mismatch: config='{cfg.embedding_model}' != manifest='{m_model}'"
            )
        if int(cfg.embedding_dimension) != m_dim:
            raise StartupIntegrityError(
                f"embedding_dimension mismatch: config={cfg.embedding_dimension} != manifest={m_dim}"
            )
        if cfg.pinecone_index_name != m_idx:
            raise StartupIntegrityError(
                f"pinecone_index_name mismatch: config='{cfg.pinecone_index_name}' != manifest='{m_idx}'"
            )
        if cfg.pinecone_namespace != m_ns:
            raise StartupIntegrityError(
                f"pinecone_namespace mismatch: config='{cfg.pinecone_namespace}' != manifest='{m_ns}'"
            )

        if snap_p.exists():
            snap_ver = read_snapshot_version(snap_p)
            if snap_ver != m_ver:
                raise StartupIntegrityError(
                    f"dataset_version mismatch: snapshot='{snap_ver}' != manifest='{m_ver}'"
                )
            exp_hash = manifest.get("snapshot_sha256")
            if exp_hash:
                actual_hash = hashlib.sha256(snap_p.read_bytes()).hexdigest()
                if actual_hash != exp_hash:
                    raise StartupIntegrityError(
                        f"snapshot_sha256 mismatch: snapshot='{actual_hash}' != manifest='{exp_hash}'"
                    )

        if led_p.exists():
            ledger = json.loads(led_p.read_text(encoding="utf-8"))
            if ledger.get("dataset_version") != m_ver:
                raise StartupIntegrityError(
                    f"dataset_version mismatch: ledger='{ledger.get('dataset_version')}' != manifest='{m_ver}'"
                )
            if ledger.get("embedding_model") != m_model:
                raise StartupIntegrityError(
                    f"embedding_model mismatch: ledger='{ledger.get('embedding_model')}' != manifest='{m_model}'"
                )
            if ledger.get("pinecone_namespace") != m_ns:
                raise StartupIntegrityError(
                    f"pinecone_namespace mismatch: ledger='{ledger.get('pinecone_namespace')}' != manifest='{m_ns}'"
                )
        elif require_ingested_ledger:
            raise StartupIntegrityError(f"Ingestion ledger missing: {led_p}")

        should_check_remote = (
            verify_remote
            if verify_remote is not None
            else (cfg.production or pinecone_stats is not None or index_description is not None or bool(cfg.pinecone_api_key))
        )
        remote_info: dict[str, Any] = {}
        if should_check_remote:
            remote_info = _verify_live_pinecone(
                cfg,
                manifest,
                pinecone_stats=pinecone_stats,
                index_description=index_description,
            )

        return {
            "status": "ok",
            "dataset_version": m_ver,
            "embedding_provider": m_provider,
            "embedding_model": m_model,
            "embedding_dimension": str(m_dim),
            "pinecone_namespace": m_ns,
            "record_count": int(manifest["record_count"]),
            **remote_info,
        }

    # Legacy fallback when manifest is omitted in local dev tests only
    snap_version = read_snapshot_version(snap_p)
    ledger = json.loads(led_p.read_text(encoding="utf-8"))
    if ledger.get("dataset_version") != snap_version:
        raise StartupIntegrityError(
            f"dataset_version mismatch: snapshot='{snap_version}' != ledger='{ledger.get('dataset_version')}'"
        )
    if ledger.get("embedding_model") != cfg.embedding_model:
        raise StartupIntegrityError(
            f"embedding_model mismatch: config='{cfg.embedding_model}' != ledger='{ledger.get('embedding_model')}'"
        )
    if ledger.get("pinecone_namespace") != cfg.pinecone_namespace:
        raise StartupIntegrityError(
            f"pinecone_namespace mismatch: config='{cfg.pinecone_namespace}' != ledger='{ledger.get('pinecone_namespace')}'"
        )
    ledger_dim = ledger.get("embedding_dimension")
    if ledger_dim is not None and int(ledger_dim) != int(cfg.embedding_dimension):
        raise StartupIntegrityError(
            f"embedding_dimension mismatch: config={cfg.embedding_dimension} != ledger={ledger_dim}"
        )
    return {
        "status": "ok",
        "dataset_version": snap_version,
        "embedding_model": str(ledger.get("embedding_model")),
        "pinecone_namespace": str(ledger.get("pinecone_namespace")),
        "embedding_dimension": str(ledger_dim),
    }
