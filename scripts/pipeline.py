#!/usr/bin/env python3
"""
DisasterMind AI — Full Pipeline
================================
1. Generate embeddings for ALL disaster records (NVIDIA nv-embed-v1, 4096-dim)
2. Apply pgvector schema to Supabase (direct Postgres)
3. Upload all embedded records to Supabase (REST API with batching + retries)

Usage:
    python scripts/full_pipeline.py                 # run all steps
    python scripts/full_pipeline.py --step embed    # only embed
    python scripts/full_pipeline.py --step schema   # only apply schema
    python scripts/full_pipeline.py --step upload   # only upload
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "disastermind_curated_dataset.json"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
OUTPUT_PATH = EMBEDDINGS_DIR / "embedded_disasters_output.json"
METADATA_PATH = EMBEDDINGS_DIR / "embedding_metadata.json"
STATS_PATH = EMBEDDINGS_DIR / "embedding_stats.json"
LOGS_DIR = PROJECT_ROOT / "logs"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 4096
EMBEDDING_MODEL = "nvidia/nv-embed-v1"
EMBED_BATCH_SIZE = 16          # NVIDIA API limit friendly
UPLOAD_BATCH_SIZE = 50         # Supabase REST payload size
MAX_RETRIES = 5

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "").strip()
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").strip()
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
SUPABASE_DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD", "").strip()


# ============================================================================
# Utilities
# ============================================================================

def _safe_str(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "unknown"
    text = str(value).strip()
    return text if text and text.lower() != "nan" else "unknown"


def _safe_number(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "unknown"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value == int(value):
            return str(int(value))
        return str(value)
    return _safe_str(value)


def _clean_number(value: Any) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if value != value:  # NaN check
            return None
        if isinstance(value, float) and math.isinf(value):
            return None
        return value
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed or math.isinf(parsed):
        return None
    return int(parsed) if parsed == int(parsed) else parsed


def _sanitize_record(record: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            clean[key] = None
        else:
            clean[key] = value
    return clean


def build_search_text(record: dict[str, Any]) -> str:
    event = _safe_str(record.get("event_name"))
    if event == "unknown":
        event = _safe_str(record.get("disaster_id"))
    return (
        f"Event: {event}\n"
        f"Disaster Type: {_safe_str(record.get('disaster_type'))}\n"
        f"Disaster Subtype: {_safe_str(record.get('disaster_subtype'))}\n"
        f"Country: {_safe_str(record.get('country'))}\n"
        f"Region: {_safe_str(record.get('region'))}\n"
        f"Location: {_safe_str(record.get('location'))}\n"
        f"Year: {_safe_number(record.get('start_year'))}\n"
        f"Deaths: {_safe_number(record.get('total_deaths'))}\n"
        f"Affected Population: {_safe_number(record.get('total_affected'))}\n"
        f"Damage USD Thousands: {_safe_number(record.get('total_damage_usd_thousands'))}\n"
        f"Magnitude: {_safe_number(record.get('magnitude'))} "
        f"{_safe_str(record.get('magnitude_scale'))}"
    )


# ============================================================================
# STEP 1: Generate Embeddings
# ============================================================================

def step_embed() -> None:
    print("\n" + "=" * 70)
    print("STEP 1: GENERATE EMBEDDINGS")
    print("=" * 70)

    if not NVIDIA_API_KEY:
        raise SystemExit("ERROR: NVIDIA_API_KEY is not set in .env")

    from openai import OpenAI

    client = OpenAI(
        api_key=NVIDIA_API_KEY,
        base_url=NVIDIA_BASE_URL,
        timeout=120,
        max_retries=2,
    )

    # Load dataset
    print(f"Loading dataset from {DATASET_PATH}...")
    with DATASET_PATH.open(encoding="utf-8") as f:
        raw_records = json.load(f)
    records = [_sanitize_record(r) for r in raw_records]
    print(f"Dataset loaded: {len(records)} records")

    # Load existing embeddings for resume
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict[str, Any]] = {}
    if OUTPUT_PATH.exists():
        try:
            with OUTPUT_PATH.open(encoding="utf-8") as f:
                prev = json.load(f)
            for row in prev:
                vec = row.get("embedding")
                if isinstance(vec, list) and len(vec) == EMBEDDING_DIM:
                    existing[str(row["id"])] = row
            print(f"Resuming: {len(existing)} records already embedded")
        except Exception:
            print("Could not load previous embeddings, starting fresh")

    embedded_rows: list[dict[str, Any]] = list(existing.values())

    # Build pending list
    pending: list[dict[str, Any]] = []
    for record in records:
        record_id = str(record.get("disaster_id") or record.get("id") or "")
        if not record_id:
            continue
        if record_id in existing:
            continue
        search_text = build_search_text(record)
        pending.append({
            **record,
            "id": record_id,
            "total_damage": record.get("total_damage_usd_thousands"),
            "search_text": search_text,
        })

    # Limit to 12000 total records
    MAX_TOTAL_RECORDS = 12000
    if len(existing) + len(pending) > MAX_TOTAL_RECORDS:
        pending = pending[:max(0, MAX_TOTAL_RECORDS - len(existing))]

    total = len(existing) + len(pending)
    print(f"Model: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"Already embedded: {len(existing)}")
    print(f"Remaining to embed: {len(pending)}")
    print(f"Total target: {total}")

    if not pending:
        print("All records already embedded!")
    else:
        for start in range(0, len(pending), EMBED_BATCH_SIZE):
            batch = pending[start:start + EMBED_BATCH_SIZE]
            texts = [row["search_text"] for row in batch]

            # Retry loop
            vectors = None
            for attempt in range(MAX_RETRIES):
                try:
                    response = client.embeddings.create(
                        model=EMBEDDING_MODEL,
                        input=texts,
                        extra_body={
                            "input_type": "passage",
                            "encoding_format": "float",
                            "truncate": "END",
                        },
                    )
                    vectors = [item.embedding for item in response.data]
                    break
                except Exception as exc:
                    if attempt == MAX_RETRIES - 1:
                        print(f"FATAL: Batch {start}-{start+len(batch)} failed after {MAX_RETRIES} retries: {exc}")
                        raise
                    wait = min(2 ** attempt, 30)
                    print(f"  Retry {attempt+1}/{MAX_RETRIES} in {wait}s ({exc})")
                    time.sleep(wait)

            for row, vector in zip(batch, vectors, strict=True):
                if len(vector) != EMBEDDING_DIM:
                    raise ValueError(f"Expected dim {EMBEDDING_DIM}, got {len(vector)} for {row['id']}")
                row["embedding"] = vector
                embedded_rows.append(row)

            done = len(existing) + start + len(batch)
            pct = round(done / total * 100, 1) if total else 100
            print(f"  Embedded {done}/{total} ({pct}%)")

            # Save checkpoint periodically to reduce I/O and Windows file lock issues
            if (start // EMBED_BATCH_SIZE) % 10 == 0 or done == total:
                tmp_path = OUTPUT_PATH.with_suffix(".json.tmp")
                with tmp_path.open("w", encoding="utf-8") as f:
                    json.dump(embedded_rows, f)
                for attempt in range(30):
                    try:
                        tmp_path.replace(OUTPUT_PATH)
                        break
                    except PermissionError:
                        if attempt == 29:
                            raise
                        time.sleep(1)

    # Write metadata
    metadata = {
        "provider": "nvidia-build",
        "model": EMBEDDING_MODEL,
        "dimension": EMBEDDING_DIM,
        "dataset_path": str(DATASET_PATH),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "record_count": len(embedded_rows),
        "dataset_record_count": len(records),
    }
    stats = {
        "dataset_total_records": len(records),
        "total_records": len(embedded_rows),
        "with_embeddings": sum(1 for r in embedded_rows if r.get("embedding")),
        "missing_embeddings": len(records) - len(embedded_rows),
        "countries": len({r.get("country") for r in embedded_rows if r.get("country")}),
        "disaster_types": len({r.get("disaster_type") for r in embedded_rows if r.get("disaster_type")}),
    }
    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with STATS_PATH.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nEmbedding complete: {len(embedded_rows)} records saved to {OUTPUT_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    print(f"Stats: {STATS_PATH}")


# ============================================================================
# STEP 2: Apply Schema
# ============================================================================

def step_schema() -> None:
    print("\n" + "=" * 70)
    print("STEP 2: APPLY SUPABASE SCHEMA")
    print("=" * 70)

    if not SUPABASE_DB_PASSWORD and not os.getenv("DATABASE_URL"):
        raise SystemExit("ERROR: SUPABASE_DB_PASSWORD or DATABASE_URL required")

    import psycopg2

    db_url = os.getenv("DATABASE_URL", "").strip()
    if db_url:
        conn = psycopg2.connect(db_url)
    else:
        project_ref = SUPABASE_URL.replace("https://", "").split(".")[0]
        host = f"db.{project_ref}.supabase.co"
        print(f"Connecting to {host}...")
        conn = psycopg2.connect(
            host=host,
            port=5432,
            dbname="postgres",
            user="postgres",
            password=SUPABASE_DB_PASSWORD,
            sslmode="require",
        )

    conn.autocommit = True
    statements = [
        "CREATE EXTENSION IF NOT EXISTS vector",
        "DROP FUNCTION IF EXISTS match_disasters(vector, integer, text, text) CASCADE",
        "DROP FUNCTION IF EXISTS match_disasters(vector, integer, double precision, text, text) CASCADE",
        "DROP TABLE IF EXISTS disaster_embeddings CASCADE",
        f"""
        CREATE TABLE disaster_embeddings (
            id TEXT PRIMARY KEY,
            event_name TEXT,
            disaster_type TEXT,
            disaster_subtype TEXT,
            country TEXT,
            region TEXT,
            location TEXT,
            start_year INTEGER,
            total_deaths DOUBLE PRECISION,
            total_affected DOUBLE PRECISION,
            total_damage DOUBLE PRECISION,
            search_text TEXT NOT NULL,
            embedding vector({EMBEDDING_DIM}) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        f"ALTER TABLE disaster_embeddings ALTER COLUMN embedding TYPE vector({EMBEDDING_DIM})",
        "CREATE INDEX IF NOT EXISTS disaster_embeddings_disaster_type_idx ON disaster_embeddings (disaster_type)",
        "CREATE INDEX IF NOT EXISTS disaster_embeddings_disaster_subtype_idx ON disaster_embeddings (disaster_subtype)",
        "CREATE INDEX IF NOT EXISTS disaster_embeddings_country_idx ON disaster_embeddings (country)",
        "CREATE INDEX IF NOT EXISTS disaster_embeddings_start_year_idx ON disaster_embeddings (start_year)",
        f"""
        CREATE OR REPLACE FUNCTION match_disasters(
            query_embedding vector({EMBEDDING_DIM}),
            match_count INTEGER DEFAULT 10,
            match_threshold DOUBLE PRECISION DEFAULT 0.0,
            filter_disaster_type TEXT DEFAULT NULL,
            filter_country TEXT DEFAULT NULL
        )
        RETURNS TABLE (
            id TEXT,
            event_name TEXT,
            disaster_type TEXT,
            disaster_subtype TEXT,
            country TEXT,
            region TEXT,
            location TEXT,
            start_year INTEGER,
            total_deaths DOUBLE PRECISION,
            total_affected DOUBLE PRECISION,
            total_damage DOUBLE PRECISION,
            search_text TEXT,
            similarity DOUBLE PRECISION
        )
        LANGUAGE sql
        STABLE
        AS $$
            WITH ranked AS (
                SELECT
                    de.id,
                    de.event_name,
                    de.disaster_type,
                    de.disaster_subtype,
                    de.country,
                    de.region,
                    de.location,
                    de.start_year,
                    de.total_deaths,
                    de.total_affected,
                    de.total_damage,
                    de.search_text,
                    (1 - (de.embedding <=> query_embedding))::DOUBLE PRECISION AS similarity,
                    de.embedding <=> query_embedding AS distance
                FROM disaster_embeddings de
                WHERE
                    (
                        filter_disaster_type IS NULL
                        OR de.disaster_type ILIKE '%' || filter_disaster_type || '%'
                        OR de.disaster_subtype ILIKE '%' || filter_disaster_type || '%'
                        OR de.search_text ILIKE '%' || filter_disaster_type || '%'
                    )
                    AND (
                        filter_country IS NULL
                        OR de.country ILIKE '%' || filter_country || '%'
                    )
            )
            SELECT
                ranked.id,
                ranked.event_name,
                ranked.disaster_type,
                ranked.disaster_subtype,
                ranked.country,
                ranked.region,
                ranked.location,
                ranked.start_year,
                ranked.total_deaths,
                ranked.total_affected,
                ranked.total_damage,
                ranked.search_text,
                ranked.similarity
            FROM ranked
            WHERE ranked.similarity >= match_threshold
            ORDER BY ranked.distance
            LIMIT LEAST(GREATEST(match_count, 1), 200)
        $$
        """,
        f"GRANT EXECUTE ON FUNCTION match_disasters(vector, integer, double precision, text, text) TO anon, authenticated, service_role",
        "GRANT SELECT, INSERT, UPDATE ON disaster_embeddings TO service_role",
        "GRANT SELECT ON disaster_embeddings TO authenticated",
        "ALTER TABLE disaster_embeddings ENABLE ROW LEVEL SECURITY",
        """
        DO $$ BEGIN
            CREATE POLICY "service_role_read" ON disaster_embeddings FOR SELECT TO service_role USING (true);
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$
        """,
        """
        DO $$ BEGIN
            CREATE POLICY "authenticated_read" ON disaster_embeddings FOR SELECT TO authenticated USING (true);
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$
        """,
        """
        DO $$ BEGIN
            CREATE POLICY "service_role_write" ON disaster_embeddings FOR ALL TO service_role USING (true) WITH CHECK (true);
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$
        """,
    ]

    try:
        with conn.cursor() as cur:
            for stmt in statements:
                try:
                    cur.execute(stmt)
                    label = stmt.strip().split('\n')[0][:80]
                    print(f"  OK: {label}")
                except Exception as exc:
                    print(f"  WARN: {stmt.strip().split(chr(10))[0][:80]} — {exc}")
    finally:
        conn.close()

    print(f"\nSchema applied (vector dim={EMBEDDING_DIM})")


# ============================================================================
# STEP 3: Upload to Supabase
# ============================================================================

def step_upload() -> None:
    print("\n" + "=" * 70)
    print("STEP 3: UPLOAD EMBEDDINGS TO SUPABASE")
    print("=" * 70)

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise SystemExit("ERROR: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY required")

    if not OUTPUT_PATH.exists():
        raise SystemExit(f"ERROR: {OUTPUT_PATH} not found. Run --step embed first.")

    import httpx

    print(f"Loading embeddings from {OUTPUT_PATH}...")
    with OUTPUT_PATH.open(encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list) or not records:
        raise SystemExit("ERROR: embedded_disasters.json is empty.")

    # Validate and build payloads
    payloads: list[dict[str, Any]] = []
    skipped = 0
    for record in records:
        vec = record.get("embedding")
        if not isinstance(vec, list) or len(vec) != EMBEDDING_DIM:
            skipped += 1
            continue
        record_id = str(record.get("id") or record.get("disaster_id") or "").strip()
        if not record_id:
            skipped += 1
            continue
        payloads.append({
            "id": record_id,
            "event_name": record.get("event_name"),
            "disaster_type": record.get("disaster_type"),
            "disaster_subtype": record.get("disaster_subtype"),
            "country": record.get("country"),
            "region": record.get("region"),
            "location": record.get("location"),
            "start_year": _clean_number(record.get("start_year")),
            "total_deaths": _clean_number(record.get("total_deaths")),
            "total_affected": _clean_number(record.get("total_affected")),
            "total_damage": _clean_number(record.get("total_damage") or record.get("total_damage_usd_thousands")),
            "search_text": record["search_text"],
            "embedding": vec,
        })

    print(f"Valid payloads: {len(payloads)} (skipped: {skipped})")

    url = SUPABASE_URL.rstrip("/")
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }
    endpoint = f"{url}/rest/v1/disaster_embeddings?on_conflict=id"

    uploaded = 0
    failed = 0
    failures: list[dict[str, Any]] = []

    with httpx.Client(timeout=300.0) as http:
        BATCH_SIZE = 100
        for start in range(0, len(payloads), BATCH_SIZE):
            batch = payloads[start:start + BATCH_SIZE]
            batch_ids = [row["id"] for row in batch]
            success = False

            for attempt in range(MAX_RETRIES):
                try:
                    response = http.post(endpoint, headers=headers, json=batch)
                    if response.status_code >= 400:
                        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
                    uploaded += len(batch)
                    pct = round(uploaded / len(payloads) * 100, 1)
                    print(f"  Uploaded {uploaded}/{len(payloads)} ({pct}%)")
                    success = True
                    break
                except Exception as exc:
                    if attempt == MAX_RETRIES - 1:
                        failed += len(batch)
                        failures.append({"ids": batch_ids, "error": str(exc)})
                        print(f"  FAILED batch {start}-{start+len(batch)}: {exc}")
                    else:
                        wait = min(2 ** attempt, 30)
                        print(f"  Retry {attempt+1}/{MAX_RETRIES} in {wait}s ({exc})")
                        time.sleep(wait)

        # Verify row count
        try:
            count_resp = http.head(
                f"{url}/rest/v1/disaster_embeddings?select=id",
                headers={**headers, "Prefer": "count=exact"},
            )
            db_count = 0
            if "content-range" in count_resp.headers:
                db_count = int(count_resp.headers["content-range"].split("/")[-1])
        except Exception:
            db_count = -1

    LOGS_DIR.mkdir(exist_ok=True)
    if failures:
        (LOGS_DIR / "upload_failures.json").write_text(
            json.dumps(failures, indent=2), encoding="utf-8"
        )

    report = (
        f"\n{'='*70}\n"
        f"UPLOAD REPORT\n"
        f"{'='*70}\n"
        f"Timestamp:       {datetime.now(timezone.utc).isoformat()}\n"
        f"Dimension:       {EMBEDDING_DIM}\n"
        f"Total Payloads:  {len(payloads)}\n"
        f"Uploaded:        {uploaded}\n"
        f"Failed:          {failed}\n"
        f"DB Row Count:    {db_count}\n"
        f"Success:         {failed == 0 and db_count >= len(payloads)}\n"
    )
    if failures:
        report += f"Failure log:     logs/upload_failures.json\n"

    report_path = PROJECT_ROOT / "upload_report.md"
    report_path.write_text(f"# Upload Report\n\n```\n{report}\n```\n", encoding="utf-8")
    print(report)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="DisasterMind AI — Full Pipeline")
    parser.add_argument("--step", choices=["embed", "schema", "upload", "all"], default="all")
    args = parser.parse_args()

    print("=" * 70)
    print("DisasterMind AI — Full Embedding + Upload Pipeline")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Model: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print("=" * 70)

    if args.step in ("all", "embed"):
        step_embed()

    if args.step in ("all", "schema"):
        step_schema()

    if args.step in ("all", "upload"):
        step_upload()

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
