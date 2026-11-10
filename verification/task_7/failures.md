# Task 7 Failures & Root-Cause Log (`R4`)

## Attempt 1 (`task-1895`, committed `7506092`)
- **Outcome:** `pytest` inside `scripts/run_gates.py` hung at `tests/test_task4_orchestration.py::test_task4_fastapi_sse_endpoint_streams_real_events` (`task-1895` & `task-1907` killed; diagnosed via 8s `faulthandler` traceback).
- **Root Cause:** In `RequestSizeLimitMiddleware._replay_receive()`, after draining the chunked request body, repeated calls returned `{"type": "http.request", "body": b"", "more_body": False}` instead of delegating to `await receive()`. Starlette `StreamingResponse.listen_for_disconnect()` loops awaiting `receive()` until `http.disconnect`, causing an infinite non-disconnect spin inside SSE responses.
- **Fix Applied:** Changed fallback return in `_replay_receive()` to `return await receive()`, cleanly allowing `listen_for_disconnect()` to await genuine disconnect events while still counting streamed body bytes up to `max_bytes`.

## Attempt 2 (live closeout on `cad7664`)
- **Outcome:** Offline 20-request check passed (`max_lag_ms=12.123`, all HTTP 200); live lifespan initialized, `/health` returned 200, and `/orchestrate-intake` returned HTTP 200 with a partial briefing after a NIM timeout retry and a subsequent rate-limit retry (`6` LLM calls). The verifier was stopped before its SSE endpoint check completed, so Task 7 is not accepted.
- **Evidence:** `verification/task_7/live_run.txt`; no complete `resilience_report.json` was produced.
- **Blocker:** Upstream NIM timeout/rate limiting prevented reliable completion of the live proof in this run. Retry only within the pre-registered 300-call cap after service availability allows a complete run.

