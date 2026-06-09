from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from backend.app.agents.base import AgentContext, AgentRunner, compute_confidence, evidence
from backend.app.schemas import AgentResult

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models" / "xgboost"


def _load_model(name: str):
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def _feature_vector(scenario, risk_output: dict[str, Any]) -> np.ndarray:
    return np.array([[
        scenario.wind_speed,
        scenario.rainfall,
        scenario.population_density,
        scenario.elevation,
        scenario.humidity,
        risk_output.get("risk_score", 0),
        scenario.population,
        scenario.hospital_capacity,
        scenario.shelter_capacity,
    ]])


class PredictionAgent(AgentRunner):
    agent_name = "prediction"

    async def run(self, ctx: AgentContext) -> AgentResult:
        scenario = ctx.scenario
        risk = ctx.outputs.get("risk", {})
        rag = ctx.rag

        if rag.get("status") != "ok" or not risk:
            return self._result(
                scenario,
                confidence=0.0,
                reasoning_summary="Prediction requires risk output and retrieved historical evidence.",
                evidence_items=[],
                recommendations=[],
                next_actions=[],
                status="failed",
                output={"status": "insufficient_data"},
                execution_time_ms=0,
            )

        affected_model = _load_model("affected_population")
        deaths_model = _load_model("fatality")
        damage_model = _load_model("economic_damage")

        if not all([affected_model, deaths_model, damage_model]):
            return self._result(
                scenario,
                confidence=0.0,
                reasoning_summary="XGBoost model artifacts not found. Run models/train_models.py first.",
                evidence_items=[],
                recommendations=["Train and register XGBoost models on the curated dataset."],
                next_actions=["Execute model training pipeline."],
                status="partial",
                output={"status": "insufficient_data", "reason": "model_artifacts_missing"},
                execution_time_ms=0,
            )

        features = _feature_vector(scenario, risk)
        predicted_affected = int(max(0, round(float(affected_model.predict(features)[0]))))
        predicted_deaths = int(max(0, round(float(deaths_model.predict(features)[0]))))
        predicted_damage_k = float(max(0, damage_model.predict(features)[0]))
        predicted_damage = int(predicted_damage_k * 1000)

        metrics_path = MODELS_DIR / "training_metrics.json"
        metrics = {}
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        retrieved = rag.get("retrieved_disasters", [])
        top_sim = float(retrieved[0].get("similarity", 0)) if retrieved else 0.0
        confidence = compute_confidence(True, len(retrieved), top_sim, [])

        return self._result(
            scenario,
            confidence=confidence,
            reasoning_summary="Predictions from trained XGBoost regressors using scenario features and historical severity.",
            evidence_items=[
                evidence("xgboost", "affected_population", f"predicted={predicted_affected}"),
                evidence("xgboost", "fatality", f"predicted={predicted_deaths}"),
                evidence("xgboost", "economic_damage", f"predicted_usd={predicted_damage}"),
            ],
            recommendations=["Use predicted affected population for evacuation sizing."],
            next_actions=["Send predictions to Explainability agent."],
            status="completed",
            output={
                "predicted_affected": predicted_affected,
                "predicted_deaths": predicted_deaths,
                "predicted_damage": predicted_damage,
                "model_metrics": metrics,
                "feature_vector": features.tolist()[0],
            },
            execution_time_ms=0,
        )
