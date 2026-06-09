#!/usr/bin/env python3
"""Upload embedded disasters to Supabase with validation, batching, retries, and logs."""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import get_settings  # noqa: E402
from backend.app.services.embedding_service import EMBEDDING_DIM  # noqa: E402

EMBEDDINGS_PATH = PROJECT_ROOT / "embeddings" / "embedded_disasters.json"
BATCH_SIZE = 100
MAX_RETRIES = 4


def _clean_number(value: Any) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if value != value:
            return None
        return value
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:
        return None
    return int(parsed) if parsed.is_integer() else parsed


def _row_payload(record: dict[str, Any]) -> dict[str, Any]:
    vector = record.get("embedding")
    if not isinstance(vector, list) or len(vector) != EMBEDDING_DIM:
        raise ValueError(f"{record.get('id')}: expected embedding dimension {EMBEDDING_DIM}")

    record_id = str(record.get("id") or record.get("disaster_id") or "").strip()
    if not record_id:
        raise ValueError("record missing id/disaster_id")

    return {
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
        "embedding": vector,
    }


def _load_records() -> list[dict[str, Any]]:
    if not EMBEDDINGS_PATH.exists():
        raise SystemExit(f"Missing {EMBEDDINGS_PATH}. Run embeddings/generate_embeddings.py first.")

    with EMBEDDINGS_PATH.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise SystemExit(f"{EMBEDDINGS_PATH} is empty. Generate embeddings before uploading.")
    return records


def upload() -> None:
    settings = get_settings()
    url = settings.supabase_url.rstrip("/")
    key = settings.supabase_service_role_key
    if not url or not key:
        raise SystemExit("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY required in .env.")

    records = _load_records()
    payloads = [_row_payload(record) for record in records]

    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }
    endpoint = f"{url}/rest/v1/disaster_embeddings?on_conflict=id"

    uploaded = 0
    failed = 0
    failures: list[dict[str, Any]] = []

    with httpx.Client(timeout=180.0) as client:
        for start in range(0, len(payloads), BATCH_SIZE):
            batch = payloads[start : start + BATCH_SIZE]
            batch_ids = [row["id"] for row in batch]
            for attempt in range(MAX_RETRIES):
                try:
                    response = client.post(endpoint, headers=headers, json=batch)
                    if response.status_code >= 400:
                        raise RuntimeError(response.text[:1000])
                    uploaded += len(batch)
                    print(f"Uploaded {uploaded}/{len(payloads)}")
                    break
                except Exception as exc:
                    if attempt == MAX_RETRIES - 1:
                        failed += len(batch)
                        failures.append({"ids": batch_ids, "error": str(exc)})
                        print(f"Batch failed ({start}-{start + len(batch)}): {exc}")
                    else:
                        time.sleep(2**attempt)

        count_response = client.head(
            f"{url}/rest/v1/disaster_embeddings?select=id",
            headers={**headers, "Prefer": "count=exact"},
        )
        db_count = 0
        if "content-range" in count_response.headers:
            db_count = int(count_response.headers["content-range"].split("/")[-1])

    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    if failures:
        (logs_dir / "upload_embeddings_failures.json").write_text(
            json.dumps(failures, indent=2),
            encoding="utf-8",
        )

    report = (
        "# Upload Report\n\n"
        f"Generated At: {datetime.now(UTC).isoformat()}\n"
        f"Embedding Dimension: {EMBEDDING_DIM}\n"
        f"Total Records: {len(payloads)}\n"
        f"Uploaded: {uploaded}\n"
        f"Failed: {failed}\n"
        f"Database Row Count: {db_count}\n"
        f"Match: {failed == 0 and db_count >= len(payloads)}\n"
    )
    if failures:
        report += "\nFailure Log: logs/upload_embeddings_failures.json\n"

    report_path = PROJECT_ROOT / "upload_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    upload()
