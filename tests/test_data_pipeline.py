"""Unit tests for Task 2P data pipeline cleaning, quarantine, deduplication, and invariants D1-D5."""

from __future__ import annotations

import math
import unicodedata

from backend.app.services.data_pipeline import (
    compute_content_hash,
    validate_and_clean_record,
)


def _sample_raw(**overrides) -> dict:
    base = {
        "disaster_id": "2020-0001-IND",
        "disaster_group": "Natural",
        "disaster_subgroup": "Meteorological",
        "disaster_type": "Storm",
        "disaster_subtype": "Tropical cyclone",
        "event_name": "Cyclone Amphan",
        "country": "India",
        "region": "Asia",
        "location": "West Bengal, Odisha",
        "magnitude": 240.0,
        "magnitude_scale": "Kph",
        "start_year": 2020,
        "start_month": 5.0,
        "start_day": 20.0,
        "end_year": 2020,
        "end_month": 5.0,
        "end_day": 21.0,
        "total_deaths": 98.0,
        "injured": float("nan"),
        "affected": 18000000.0,
        "homeless": float("nan"),
        "total_affected": 18000000.0,
        "total_damage_usd_thousands": 13000000.0,
        "latitude": 21.5,
        "longitude": 88.2,
    }
    base.update(overrides)
    return base


def test_filler_literals_and_name_unknown_location_stripped():
    raw = _sample_raw(
        event_name="Unknown",
        location="Puli city (Name Unknown (13003) district, Taiwan Sheng province)",
        magnitude=float("nan"),
        magnitude_scale="Kph",
    )
    cleaned, err = validate_and_clean_record(raw)
    assert err is None and cleaned is not None
    assert cleaned["event_name"] == "Storm - India (2020)"
    assert cleaned["event_name_synthesized"] is True
    assert "unknown" not in cleaned["search_text"].lower()
    assert cleaned["magnitude_scale"] is None
    assert "location_placeholder_cleaned" in cleaned["quality_flags"]


def test_nfc_normalization_applied():
    nfd = unicodedata.normalize("NFD", "Quinindé")
    raw = _sample_raw(location=nfd)
    cleaned, err = validate_and_clean_record(raw)
    assert err is None and cleaned is not None
    assert cleaned["location"] == unicodedata.normalize("NFC", "Quinindé")


def test_quarantines_temporal_inversion_and_invalid_physics():
    # Real EM-DAT temporal inversion case (1988-0424-VEN)
    inv_rec = _sample_raw(disaster_id="1988-0424-VEN", start_year=2026, end_year=1988)
    cleaned, err = validate_and_clean_record(inv_rec)
    assert cleaned is None and err is not None
    assert err[0] == "TEMPORAL_INVERSION_END_BEFORE_START"

    # Bad coordinates
    bad_coord = _sample_raw(latitude=120.0)
    _, err_coord = validate_and_clean_record(bad_coord)
    assert err_coord is not None and err_coord[0] == "INVALID_COORDINATES"

    # Impossible earthquake magnitude
    bad_mag = _sample_raw(disaster_type="Earthquake", magnitude=11.5, magnitude_scale="Moment Magnitude")
    _, err_mag = validate_and_clean_record(bad_mag)
    assert err_mag is not None and err_mag[0] == "IMPLAUSIBLE_MAGNITUDE"

    # Negative deaths
    neg_deaths = _sample_raw(total_deaths=-5.0)
    _, err_neg = validate_and_clean_record(neg_deaths)
    assert err_neg is not None and err_neg[0] == "NEGATIVE_IMPACT_VALUE"

    # Invalid calendar day (Feb 30)
    bad_cal = _sample_raw(start_month=2, start_day=30)
    _, err_cal = validate_and_clean_record(bad_cal)
    assert err_cal is not None and err_cal[0] == "INVALID_CALENDAR_DATE"


def test_exact_content_hash_detects_duplicates():
    r1, _ = validate_and_clean_record(_sample_raw(disaster_id="2020-0001-IND"))
    r2, _ = validate_and_clean_record(_sample_raw(disaster_id="2020-9999-IND"))
    assert r1 is not None and r2 is not None
    assert compute_content_hash(r1) == compute_content_hash(r2)


def test_d4_startup_guard_refuses_mismatch(tmp_path):
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    from backend.app.config import Settings
    from backend.app.services.snapshot_guard import StartupIntegrityError, verify_runtime_alignment

    snap = tmp_path / "snapshot.parquet"
    ledger = tmp_path / "ledger.json"
    no_manifest = tmp_path / "no_manifest.json"
    schema = pa.schema(
        [pa.field("disaster_id", pa.string())],
        metadata={b"dataset_version": b"ver-abc"},
    )
    pq.write_table(pa.Table.from_pylist([{"disaster_id": "1"}], schema=schema), snap)
    ledger.write_text(
        json.dumps(
            {
                "dataset_version": "ver-abc",
                "embedding_model": "nvidia/nemotron-3.5-embed-1b",
                "pinecone_namespace": "v2-ver-abc",
                "embedding_dimension": 2048,
            }
        ),
        encoding="utf-8",
    )

    ok_cfg = Settings(
        nvidia_api_key="k",
        embedding_provider="nvidia",
        embedding_model="nvidia/nemotron-3.5-embed-1b",
        pinecone_namespace="v2-ver-abc",
        embedding_dimension=2048,
    )
    assert verify_runtime_alignment(ok_cfg, manifest_path=no_manifest, snapshot_path=snap, ledger_path=ledger)["status"] == "ok"

    bad_model = Settings(
        nvidia_api_key="k",
        embedding_provider="nvidia",
        embedding_model="other-model",
        pinecone_namespace="v2-ver-abc",
        embedding_dimension=2048,
    )
    try:
        verify_runtime_alignment(bad_model, manifest_path=no_manifest, snapshot_path=snap, ledger_path=ledger)
        raise AssertionError("Expected StartupIntegrityError on model mismatch")
    except StartupIntegrityError as exc:
        assert "embedding_model mismatch" in str(exc)

    bad_ns = Settings(
        nvidia_api_key="k",
        embedding_provider="nvidia",
        embedding_model="nvidia/nemotron-3.5-embed-1b",
        pinecone_namespace="wrong-ns",
        embedding_dimension=2048,
    )
    try:
        verify_runtime_alignment(bad_ns, manifest_path=no_manifest, snapshot_path=snap, ledger_path=ledger)
        raise AssertionError("Expected StartupIntegrityError on namespace mismatch")
    except StartupIntegrityError as exc:
        assert "pinecone_namespace mismatch" in str(exc)

    ledger_wrong_ver = tmp_path / "ledger_bad_ver.json"
    ledger_wrong_ver.write_text(
        json.dumps(
            {
                "dataset_version": "ver-different",
                "embedding_model": "nvidia/nemotron-3.5-embed-1b",
                "pinecone_namespace": "v2-ver-abc",
                "embedding_dimension": 2048,
            }
        ),
        encoding="utf-8",
    )
    try:
        verify_runtime_alignment(ok_cfg, manifest_path=no_manifest, snapshot_path=snap, ledger_path=ledger_wrong_ver)
        raise AssertionError("Expected StartupIntegrityError on dataset_version mismatch")
    except StartupIntegrityError as exc:
        assert "dataset_version mismatch" in str(exc)


def test_d4_fastapi_app_refuses_startup_on_mismatch(tmp_path, monkeypatch):
    """Prove the FastAPI app itself refuses to boot on dataset_version, model, or namespace mismatch."""
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    from fastapi.testclient import TestClient

    from backend.app.config import Settings
    from backend.app.main import app
    from backend.app.services import snapshot_guard
    from backend.app.services.snapshot_guard import StartupIntegrityError

    snap = tmp_path / "snapshot.parquet"
    ledger = tmp_path / "ledger.json"
    manifest = tmp_path / "missing_manifest.json"
    schema = pa.schema([pa.field("disaster_id", pa.string())], metadata={b"dataset_version": b"ver-live"})
    pq.write_table(pa.Table.from_pylist([{"disaster_id": "1"}], schema=schema), snap)
    ledger.write_text(
        json.dumps(
            {
                "dataset_version": "ver-live",
                "embedding_model": "llama-text-embed-v2",
                "pinecone_namespace": "v2-ver-live-llama-text-embed-v2",
                "embedding_dimension": 1024,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(snapshot_guard, "DEFAULT_MANIFEST_PATH", manifest)
    monkeypatch.setattr(snapshot_guard, "DEFAULT_SNAPSHOT_PATH", snap)
    monkeypatch.setattr(snapshot_guard, "DEFAULT_LEDGER_PATH", ledger)

    # 1. Valid alignment -> FastAPI boots and /health responds 200
    valid_cfg = Settings(
        nvidia_api_key="test-key",
        embedding_provider="pinecone",
        embedding_model="llama-text-embed-v2",
        embedding_dimension=1024,
        pinecone_namespace="v2-ver-live-llama-text-embed-v2",
        enable_backend_keepalive=False,
    )
    monkeypatch.setattr("backend.app.main.get_settings", lambda: valid_cfg)
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200

    # 2. Mismatched model / namespace / dataset_version -> FastAPI startup refuses
    for bad_cfg in [
        Settings(nvidia_api_key="test-key", embedding_provider="pinecone", embedding_model="wrong-model", embedding_dimension=1024, pinecone_namespace="v2-ver-live-llama-text-embed-v2", enable_backend_keepalive=False),
        Settings(nvidia_api_key="test-key", embedding_provider="pinecone", embedding_model="llama-text-embed-v2", embedding_dimension=1024, pinecone_namespace="wrong-ns", enable_backend_keepalive=False),
    ]:
        monkeypatch.setattr("backend.app.main.get_settings", lambda cfg=bad_cfg: cfg)
        try:
            with TestClient(app):
                raise AssertionError("Expected FastAPI startup to abort on integrity mismatch")
        except StartupIntegrityError:
            pass

    ledger.write_text(
        json.dumps(
            {
                "dataset_version": "ver-stale",
                "embedding_model": "llama-text-embed-v2",
                "pinecone_namespace": "v2-ver-live-llama-text-embed-v2",
                "embedding_dimension": 1024,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("backend.app.main.get_settings", lambda: valid_cfg)
    try:
        with TestClient(app):
            raise AssertionError("Expected FastAPI startup to abort on dataset_version mismatch")
    except StartupIntegrityError:
        pass


def test_a3_deployment_manifest_guard_all_failure_modes(tmp_path):
    """A3: Manifest + Pinecone remote guard refuses missing manifest in prod, deleted-ledger bypass, and all mismatches."""
    import json
    import pytest
    from backend.app.config import Settings
    from backend.app.services.snapshot_guard import StartupIntegrityError, verify_runtime_alignment

    manifest_path = tmp_path / "index_manifest.json"
    missing_manifest = tmp_path / "does_not_exist.json"
    absent_ledger = tmp_path / "deleted_ledger.json"
    absent_snapshot = tmp_path / "absent_snapshot.parquet"

    manifest = {
        "dataset_version": "3d35e5131d79a79d",
        "snapshot_sha256": "deadbeef",
        "record_count": 16833,
        "embedding_provider": "pinecone",
        "embedding_model": "llama-text-embed-v2",
        "embedding_dimension": 1024,
        "pinecone_index_name": "disastermind",
        "pinecone_namespace": "v2-3d35e5131d79a79d-llama-text-embed-v2",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    prod_cfg = Settings(
        nvidia_api_key="k",
        pinecone_api_key="pcsk_dummy",
        embedding_provider="pinecone",
        embedding_model="llama-text-embed-v2",
        embedding_dimension=1024,
        pinecone_index_name="disastermind",
        pinecone_namespace="v2-3d35e5131d79a79d-llama-text-embed-v2",
        app_env="production",
    )
    ok_desc = {"name": "disastermind", "dimension": 1024, "metric": "cosine"}
    ok_stats = {"namespaces": {"v2-3d35e5131d79a79d-llama-text-embed-v2": {"vector_count": 16833}}}

    # Even with ledger and snapshot deleted (deployment container), valid manifest + Pinecone passes
    res = verify_runtime_alignment(
        prod_cfg,
        manifest_path=manifest_path,
        snapshot_path=absent_snapshot,
        ledger_path=absent_ledger,
        pinecone_stats=ok_stats,
        index_description=ok_desc,
    )
    assert res["status"] == "ok" and res["remote_vector_count"] == 16833

    # Missing manifest in production refuses even with deleted ledger
    with pytest.raises(StartupIntegrityError, match="Committed index manifest missing"):
        verify_runtime_alignment(
            prod_cfg,
            manifest_path=missing_manifest,
            snapshot_path=absent_snapshot,
            ledger_path=absent_ledger,
        )

    # Every config vs manifest mismatch refuses
    mismatch_cases = [
        ("embedding_provider", Settings(nvidia_api_key="k", pinecone_api_key="p", embedding_provider="nvidia", embedding_model="llama-text-embed-v2", embedding_dimension=1024, pinecone_namespace="v2-3d35e5131d79a79d-llama-text-embed-v2", app_env="production")),
        ("embedding_model", Settings(nvidia_api_key="k", pinecone_api_key="p", embedding_provider="pinecone", embedding_model="multilingual-e5-large", embedding_dimension=1024, pinecone_namespace="v2-3d35e5131d79a79d-llama-text-embed-v2", app_env="production")),
        ("embedding_dimension", Settings(nvidia_api_key="k", pinecone_api_key="p", embedding_provider="pinecone", embedding_model="llama-text-embed-v2", embedding_dimension=2048, pinecone_namespace="v2-3d35e5131d79a79d-llama-text-embed-v2", app_env="production")),
        ("pinecone_index_name", Settings(nvidia_api_key="k", pinecone_api_key="p", embedding_provider="pinecone", embedding_model="llama-text-embed-v2", embedding_dimension=1024, pinecone_index_name="other-index", pinecone_namespace="v2-3d35e5131d79a79d-llama-text-embed-v2", app_env="production")),
        ("pinecone_namespace", Settings(nvidia_api_key="k", pinecone_api_key="p", embedding_provider="pinecone", embedding_model="llama-text-embed-v2", embedding_dimension=1024, pinecone_namespace="wrong-ns", app_env="production")),
    ]
    for pattern, bad_cfg in mismatch_cases:
        with pytest.raises(StartupIntegrityError, match=pattern):
            verify_runtime_alignment(
                bad_cfg,
                manifest_path=manifest_path,
                snapshot_path=absent_snapshot,
                ledger_path=absent_ledger,
                pinecone_stats=ok_stats,
                index_description=ok_desc,
            )

    # Remote Pinecone mismatches refuse even if config == manifest and ledger is deleted
    with pytest.raises(StartupIntegrityError, match="dimension mismatch"):
        verify_runtime_alignment(
            prod_cfg,
            manifest_path=manifest_path,
            snapshot_path=absent_snapshot,
            ledger_path=absent_ledger,
            pinecone_stats=ok_stats,
            index_description={"name": "disastermind", "dimension": 2048},
        )
    with pytest.raises(StartupIntegrityError, match="namespace missing"):
        verify_runtime_alignment(
            prod_cfg,
            manifest_path=manifest_path,
            snapshot_path=absent_snapshot,
            ledger_path=absent_ledger,
            pinecone_stats={"namespaces": {}},
            index_description=ok_desc,
        )
    with pytest.raises(StartupIntegrityError, match="vector count mismatch"):
        verify_runtime_alignment(
            prod_cfg,
            manifest_path=manifest_path,
            snapshot_path=absent_snapshot,
            ledger_path=absent_ledger,
            pinecone_stats={"namespaces": {"v2-3d35e5131d79a79d-llama-text-embed-v2": {"vector_count": 16828}}},
            index_description=ok_desc,
        )


def test_a8_quarantine_source_1988_0424_ven_regression():
    """A8 regression: 1988-0424-VEN source temporal inversion (2026 > 1988) is quarantined, never guessed."""
    raw = _sample_raw(
        disaster_id="1988-0424-VEN",
        event_name="Hurricane 'Gilbert'",
        country="Venezuela (Bolivarian Republic of)",
        start_year=2026,
        start_month=9.0,
        start_day=11.0,
        end_year=1988,
        end_month=9.0,
        end_day=11.0,
        total_deaths=5.0,
        total_affected=float("nan"),
        affected=float("nan"),
        total_damage_usd_thousands=3000.0,
    )
    cleaned, err = validate_and_clean_record(raw)
    assert cleaned is None and err is not None
    assert err[0] == "TEMPORAL_INVERSION_END_BEFORE_START"
    assert "2026" in err[1] and "1988" in err[1]


import pytest


@pytest.mark.live
def test_live_deployment_manifest_and_pinecone_alignment():
    from backend.app.config import get_settings
    from backend.app.services.snapshot_guard import verify_runtime_alignment

    res = verify_runtime_alignment(get_settings())
    assert res["status"] == "ok"
    assert res["remote_vector_count"] == 16833
    assert res["remote_dimension"] == 1024



