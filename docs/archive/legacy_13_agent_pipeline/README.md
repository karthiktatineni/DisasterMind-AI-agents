# Archive Note: Legacy 13-Agent Pipeline, Obsolete Tests, and Hardcoded Audit Suite

Moved out of active runtime and test paths in **Task 4** (`R3` rule: archive obsolete code and tests with an explicit note rather than deleting silently):

- `agents/*.py` & `legacy_orchestration.py`: Legacy 13-agent sequential/staged pipeline (`Coordinator`, `Knowledge`, `Similarity`, `Risk`, `Prediction` via `.joblib` XGBoost, `Explainability`, `Logistics`, `Evacuation`, `Simulation`, `Validation`, `Planning`, `Decision`, `Reporting`), superseded by the native tool-calling **4-agent** architecture (`Coordinator`, `Risk`, `Logistics`, `Evacuation`).
- `legacy_test_orchestration.py`: Obsolete tests targeting the 13-agent `AGENT_SEQUENCE` and mock XGBoost `.joblib` loaders. Replaced cleanly by native tool-calling Task 4 tests under `tests/`.
- `docs/archive/audit_test_suite.py` & `docs/archive/benchmark_latency.py`: Legacy audit/benchmark scripts archived because `audit_test_suite.py` hardcoded static/synthetic `execution_time_ms` values and targeted the 13-agent runner. Zero metrics from archived audit scripts are cited in current verification reports.
- `models/`: Removed `.joblib` serialized XGBoost artifacts, `train_models.py`, and `xgboost`/`joblib`/`scikit-learn`/`pandas` dependencies from active tracking (`requirements.txt`, `backend/requirements.txt`), eliminating the pickle deserialization surface.
- `backend/app/agents/fallbacks.py`: Deprecated heuristic fallback module (not imported by the 4-agent pipeline; scheduled for deletion and before/after grep audit in Task 5).

