"""Tests for HistoricalStatsService: n<5 guardrails, reference equivalence, property tests, and mutation tests."""

from __future__ import annotations

import math
import random
import statistics
from typing import Callable

import pytest

from backend.app.services.historical_stats import (
    MIN_SAMPLE_SIZE,
    HistoricalStatsService,
    _compute_metric_summary,
)


def _reference_linear_percentile(sorted_vals: list[float], pct: float) -> float:
    """Pure-Python textbook linear interpolation percentile (no NumPy)."""
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])
    pos = (pct / 100.0) * (n - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    frac = pos - lo
    return float(sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo]))


def _reference_summary(vals: list[float], min_n: int = 5) -> dict:
    if len(vals) < min_n:
        return {"status": "no_data", "median": None, "p90": None}
    s = sorted(vals)
    return {
        "status": "ok",
        "median": round(float(statistics.median(s)), 4),
        "p90": round(_reference_linear_percentile(s, 90.0), 4),
    }


def _make_records(deaths_list: list[float | None], dtype: str = "Flood", country: str = "India") -> list[dict]:
    return [
        {
            "disaster_id": f"2020-{i:04d}-TST",
            "disaster_type": dtype,
            "disaster_subtype": "Riverine flood",
            "country": country,
            "start_year": 2000 + (i % 25),
            "total_deaths": d,
            "total_affected": (d * 100.0) if d is not None else None,
            "total_damage_usd_thousands": (d * 10.0) if d is not None else None,
        }
        for i, d in enumerate(deaths_list)
    ]


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
def test_n_below_5_returns_no_data(n: int):
    records = _make_records([10.0 * (i + 1) for i in range(n)])
    svc = HistoricalStatsService(records=records, dataset_version="test-v1")
    res = svc.query_stats(disaster_type="Flood", country="India")
    assert res["status"] == "no_data"
    assert res["sample_size"] == n
    assert res["min_required"] == MIN_SAMPLE_SIZE
    assert res["metrics"] == {}


def test_per_metric_guardrail_when_record_count_ok_but_metric_sparse():
    # 6 matching records total, but only 3 have total_damage_usd_thousands populated
    records = _make_records([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    for r in records[:3]:
        r["total_damage_usd_thousands"] = None
    svc = HistoricalStatsService(records=records, dataset_version="test-v1")
    res = svc.query_stats(disaster_type="Flood", country="India")
    assert res["status"] == "ok"
    assert res["metrics"]["total_deaths"]["status"] == "ok"
    assert res["metrics"]["total_damage_usd_thousands"]["status"] == "no_data"
    assert res["metrics"]["total_damage_usd_thousands"]["sample_size"] == 3


def test_equivalence_against_pure_python_reference():
    rng = random.Random(12345)
    for size in [5, 6, 7, 11, 20, 53]:
        vals = [round(rng.uniform(1.0, 50000.0), 3) for _ in range(size)]
        records = _make_records(vals)
        svc = HistoricalStatsService(records=records, dataset_version="test-v1")
        res = svc.query_stats(disaster_type="Flood", country="India")
        ref = _reference_summary(vals)
        assert res["status"] == "ok"
        assert res["metrics"]["total_deaths"]["median"] == pytest.approx(ref["median"], abs=1e-4)
        assert res["metrics"]["total_deaths"]["p90"] == pytest.approx(ref["p90"], abs=1e-4)


def test_properties_permutation_scaling_and_monotonicity():
    rng = random.Random(99)
    for trial in range(15):
        vals = [round(rng.uniform(0.0, 10000.0), 2) for _ in range(rng.randint(5, 40))]
        shuffled = list(vals)
        rng.shuffle(shuffled)
        scale = 3.5
        scaled = [v * scale for v in vals]

        s1 = HistoricalStatsService(records=_make_records(vals)).query_stats(disaster_type="Flood")
        s2 = HistoricalStatsService(records=_make_records(shuffled)).query_stats(disaster_type="Flood")
        s3 = HistoricalStatsService(records=_make_records(scaled)).query_stats(disaster_type="Flood")

        m1 = s1["metrics"]["total_deaths"]
        m2 = s2["metrics"]["total_deaths"]
        m3 = s3["metrics"]["total_deaths"]

        assert m1["median"] == m2["median"]
        assert m1["p90"] == m2["p90"]
        assert m1["min"] <= m1["median"] <= m1["p90"] <= m1["max"]
        assert m3["median"] == pytest.approx(m1["median"] * scale, rel=1e-3, abs=1e-2)
        assert m3["p90"] == pytest.approx(m1["p90"] * scale, rel=1e-3, abs=1e-2)


def _verify_stat_contract(fn: Callable[[list[float]], dict]) -> None:
    # Helper assertion suite that both real implementation and mutants are tested against
    assert fn([1.0, 2.0, 3.0, 4.0])["status"] == "no_data"
    vals = [10.0, 20.0, 30.0, 40.0, 100.0]
    out = fn(vals)
    assert out["status"] == "ok"
    ref = _reference_summary(vals)
    assert out["median"] == pytest.approx(ref["median"], abs=1e-4)
    assert out["p90"] == pytest.approx(ref["p90"], abs=1e-4)


def test_mutation_resistance():
    # Real implementation must pass
    _verify_stat_contract(_compute_metric_summary)

    # Mutant 1: breaks n<5 guardrail
    def mutant_no_guardrail(vals: list[float]) -> dict:
        return _compute_metric_summary(vals, min_sample_size=1)

    # Mutant 2: returns mean instead of median
    def mutant_mean_for_median(vals: list[float]) -> dict:
        res = _compute_metric_summary(vals, min_sample_size=5)
        if res["status"] == "ok":
            res["median"] = round(sum(vals) / len(vals), 4)
        return res

    # Mutant 3: returns max instead of p90
    def mutant_max_for_p90(vals: list[float]) -> dict:
        res = _compute_metric_summary(vals, min_sample_size=5)
        if res["status"] == "ok":
            res["p90"] = max(vals)
        return res

    for mutant in (mutant_no_guardrail, mutant_mean_for_median, mutant_max_for_p90):
        with pytest.raises(AssertionError):
            _verify_stat_contract(mutant)

