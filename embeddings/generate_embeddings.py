#!/usr/bin/env python3
"""Generate embeddings for the curated disaster dataset using text-embedding-3-small."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from embeddings.build_search_text import build_search_text  # noqa: E402

DATASET_PATH = PROJECT_ROOT / "data" / "disastermind_curated_dataset.json"
OUTPUT_DIR = PROJECT_ROOT / "embeddings"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
BATCH_SIZE = 100


def _sanitize_record(record: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            clean[key] = None
        else:
            clean[key] = value
    return clean


def _load_dataset(limit: int | None = None) -> list[dict[str, Any]]:
    with DATASET_PATH.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if limit:
        return records[:limit]
    return records


def _embed_batch(client: OpenAI, texts: list[str], retries: int = 3) -> list[list[float]]:
    for attempt in range(retries):
        try:
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
            return [item.embedding for item in response.data]
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"Embedding batch failed ({exc}); retry in {wait}s...")
            time.sleep(wait)
    raise RuntimeError("Unreachable")


def generate(limit: int | None = None, resume: bool = False) -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required in .env for embedding generation.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "embedded_disasters.json"
    existing: dict[str, dict[str, Any]] = {}
    if resume and output_path.exists():
        with output_path.open(encoding="utf-8") as handle:
            for row in json.load(handle):
                existing[row["id"]] = row

    client = OpenAI(api_key=api_key)
    records = _load_dataset(limit=limit)
    embedded_rows: list[dict[str, Any]] = list(existing.values())
    pending: list[dict[str, Any]] = []

    for record in records:
        record = _sanitize_record(record)
        record_id = str(record.get("disaster_id") or record.get("id"))
        if record_id in existing:
            continue
        search_text = build_search_text(record)
        pending.append(
            {
                "id": record_id,
                "event_name": record.get("event_name"),
                "disaster_type": record.get("disaster_type"),
                "disaster_subtype": record.get("disaster_subtype"),
                "country": record.get("country"),
                "region": record.get("region"),
                "location": record.get("location"),
                "start_year": record.get("start_year"),
                "total_deaths": record.get("total_deaths"),
                "total_affected": record.get("total_affected"),
                "total_damage": record.get("total_damage_usd_thousands"),
                "search_text": search_text,
                "source_record": record,
            }
        )

    total = len(existing) + len(pending)
    print(f"Records to embed: {len(pending)} (total target: {total})")

    for start in range(0, len(pending), BATCH_SIZE):
        batch = pending[start : start + BATCH_SIZE]
        texts = [row["search_text"] for row in batch]
        vectors = _embed_batch(client, texts)
        for row, vector in zip(batch, vectors, strict=True):
            if len(vector) != EMBEDDING_DIM:
                raise ValueError(f"Expected dim {EMBEDDING_DIM}, got {len(vector)}")
            row["embedding"] = vector
            embedded_rows.append(row)
        done = len(existing) + start + len(batch)
        print(f"Embedded {done}/{total}")
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(embedded_rows, handle)

    metadata = {
        "model": EMBEDDING_MODEL,
        "dimension": EMBEDDING_DIM,
        "dataset_path": str(DATASET_PATH),
        "generated_at": datetime.now(UTC).isoformat(),
        "record_count": len(embedded_rows),
    }
    stats = {
        "total_records": len(embedded_rows),
        "with_embeddings": sum(1 for row in embedded_rows if row.get("embedding")),
        "countries": len({row.get("country") for row in embedded_rows if row.get("country")}),
        "disaster_types": len({row.get("disaster_type") for row in embedded_rows if row.get("disaster_type")}),
    }
    with (OUTPUT_DIR / "embedding_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    with (OUTPUT_DIR / "embedding_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    print(f"Wrote {output_path} ({len(embedded_rows)} records)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit records for testing")
    parser.add_argument("--resume", action="store_true", help="Resume from partial output")
    args = parser.parse_args()
    generate(limit=args.limit, resume=args.resume)
