#!/usr/bin/env python3
"""Train XGBoost regressors on the curated disaster dataset."""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "data" / "disastermind_curated_dataset.json"
OUTPUT_DIR = PROJECT_ROOT / "models" / "xgboost"

FEATURE_COLUMNS = [
    "wind_proxy",
    "rain_proxy",
    "population_density_proxy",
    "elevation_proxy",
    "humidity_proxy",
    "historical_severity_proxy",
    "population_proxy",
    "hospital_proxy",
    "shelter_proxy",
]

TARGETS = {
    "affected_population": "total_affected",
    "fatality": "total_deaths",
    "economic_damage": "total_damage_usd_thousands",
}


def _safe_float(value) -> float:
    if value is None:
        return 0.0
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return 0.0
    return float(value)


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    magnitude = df["magnitude"].apply(_safe_float)
    affected = df["total_affected"].apply(_safe_float)
    deaths = df["total_deaths"].apply(_safe_float)

    features = pd.DataFrame()
    features["wind_proxy"] = np.where(df["disaster_type"].str.contains("Storm|Cyclone", case=False, na=False), magnitude * 20, 30)
    features["rain_proxy"] = np.where(df["disaster_type"].str.contains("Flood|Storm|Cyclone", case=False, na=False), magnitude * 15, 20)
    features["population_density_proxy"] = np.log1p(affected.clip(lower=0))
    features["elevation_proxy"] = 50 - df["latitude"].apply(_safe_float).clip(0, 50)
    features["humidity_proxy"] = 60.0
    features["historical_severity_proxy"] = (deaths.clip(lower=0) / 1000).clip(0, 1)
    features["population_proxy"] = affected
    features["hospital_proxy"] = (affected * 0.026).clip(lower=250)
    features["shelter_proxy"] = (affected * 0.5).clip(lower=500)
    return features


def train() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with DATASET.open(encoding="utf-8") as handle:
        records = json.load(handle)
    df = pd.DataFrame(records)

    features = _build_features(df)
    metrics: dict = {}
    importance: dict = {}

    for model_name, target_col in TARGETS.items():
        y = df[target_col].apply(_safe_float)
        mask = y > 0
        X = features.loc[mask]
        y = y.loc[mask]
        if len(X) < 100:
            print(f"Skipping {model_name}: insufficient labeled rows ({len(X)})")
            continue

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        metrics[model_name] = {"r2_test": round(float(score), 4), "train_rows": len(X_train), "test_rows": len(X_test)}
        importance[model_name] = dict(zip(FEATURE_COLUMNS, [round(float(v), 4) for v in model.feature_importances_], strict=True))
        joblib.dump(model, OUTPUT_DIR / f"{model_name}.joblib")
        print(f"Trained {model_name}: R2={score:.4f}")

    (OUTPUT_DIR / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "feature_importance.json").write_text(json.dumps(importance, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "training_report.md").write_text(
        "# Model Training Report\n\n" + json.dumps(metrics, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    train()
