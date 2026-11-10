"""In-memory historical aggregate statistics computed from the canonical Parquet snapshot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SNAPSHOT_PATH = PROJECT_ROOT / "data" / "disastermind_snapshot.parquet"
MIN_SAMPLE_SIZE = 5

IMPACT_METRICS = (
    "total_deaths",
    "total_affected",
    "total_damage_usd_thousands",
)


def _compute_metric_summary(values: list[float], min_sample_size: int = MIN_SAMPLE_SIZE) -> dict[str, Any]:
    n = len(values)
    if n < min_sample_size:
        return {
            "status": "no_data",
            "sample_size": n,
            "min_required": min_sample_size,
            "median": None,
            "p90": None,
        }
    arr = np.sort(np.asarray(values, dtype=np.float64))
    median = float(np.median(arr))
    p90 = float(np.percentile(arr, 90, method="linear"))
    return {
        "status": "ok",
        "sample_size": n,
        "min_required": min_sample_size,
        "min": float(arr[0]),
        "median": round(median, 4),
        "p90": round(p90, 4),
        "max": float(arr[-1]),
    }


class HistoricalStatsService:
    """Queries medians and p90 impacts over the canonical disaster snapshot with n<5 guardrails."""

    def __init__(
        self,
        snapshot_path: Path | None = None,
        records: list[dict[str, Any]] | None = None,
        dataset_version: str | None = None,
        min_sample_size: int = MIN_SAMPLE_SIZE,
    ) -> None:
        self.min_sample_size = min_sample_size
        if records is not None:
            self._records = list(records)
            self.dataset_version = dataset_version or "in-memory"
            return

        path = snapshot_path or DEFAULT_SNAPSHOT_PATH
        if not path.exists():
            raise FileNotFoundError(f"Canonical snapshot not found: {path}")
        table = pq.read_table(path)
        meta = table.schema.metadata or {}
        self.dataset_version = meta.get(b"dataset_version", b"unknown").decode("utf-8")
        self._records = table.to_pylist()

    @property
    def total_records(self) -> int:
        return len(self._records)

    def query_stats(
        self,
        *,
        disaster_type: str | None = None,
        country: str | None = None,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> dict[str, Any]:
        dtype_norm = disaster_type.strip().lower() if disaster_type else None
        country_norm = country.strip().lower() if country else None

        matched: list[dict[str, Any]] = []
        for rec in self._records:
            if dtype_norm:
                rec_type = (rec.get("disaster_type") or "").strip().lower()
                rec_sub = (rec.get("disaster_subtype") or "").strip().lower()
                if rec_type != dtype_norm and rec_sub != dtype_norm:
                    continue
            if country_norm:
                rec_country = (rec.get("country") or "").strip().lower()
                if rec_country != country_norm:
                    continue
            sy = rec.get("start_year")
            if start_year is not None and (sy is None or int(sy) < start_year):
                continue
            if end_year is not None and (sy is None or int(sy) > end_year):
                continue
            matched.append(rec)

        n_matched = len(matched)
        filters = {
            "disaster_type": disaster_type,
            "country": country,
            "start_year": start_year,
            "end_year": end_year,
        }

        if n_matched < self.min_sample_size:
            return {
                "status": "no_data",
                "reason": f"Insufficient historical sample size ({n_matched} < {self.min_sample_size}).",
                "sample_size": n_matched,
                "min_required": self.min_sample_size,
                "dataset_version": self.dataset_version,
                "filters": filters,
                "metrics": {},
            }

        metrics_summary: dict[str, Any] = {}
        for metric in IMPACT_METRICS:
            vals = [
                float(r[metric])
                for r in matched
                if r.get(metric) is not None and not np.isnan(float(r[metric]))
            ]
            metrics_summary[metric] = _compute_metric_summary(vals, min_sample_size=self.min_sample_size)

        return {
            "status": "ok",
            "sample_size": n_matched,
            "min_required": self.min_sample_size,
            "dataset_version": self.dataset_version,
            "filters": filters,
            "metrics": metrics_summary,
        }

