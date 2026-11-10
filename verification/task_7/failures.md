# Task 7 Failures & Root-Cause Log (`R4`)

## Attempt 1 (`task-1895`, committed `7506092`)
- **Outcome:** `pytest` inside `scripts/run_gates.py` hung at `tests/test_task4_orchestration.py::test_task4_fastapi_sse_endpoint_streams_real_events` (`task-1895` & `task-1907` killed; diagnosed via 8s `faulthandler` traceback).
- **Root Cause:** In `RequestSizeLimitMiddleware._replay_receive()`, after draining the chunked request body, repeated calls returned `{"type": "http.request", "body": b"", "more_body": False}` instead of delegating to `await receive()`. Starlette `StreamingResponse.listen_for_disconnect()` loops awaiting `receive()` until `http.disconnect`, causing an infinite non-disconnect spin inside SSE responses.
- **Fix Applied:** Changed fallback return in `_replay_receive()` to `return await receive()`, cleanly allowing `listen_for_disconnect()` to await genuine disconnect events while still counting streamed body bytes up to `max_bytes`.

## Attempt 2 (live closeout on `cad7664`)
- **Outcome:** Offline 20-request check passed (`max_lag_ms=12.123`, all HTTP 200); live lifespan initialized, `/health` returned 200, and `/orchestrate-intake` returned HTTP 200 with a partial briefing after a NIM timeout retry and a subsequent rate-limit retry (`6` LLM calls). The verifier was stopped before its SSE endpoint check completed, so Task 7 is not accepted.
- **Evidence:** `verification/task_7/live_run.txt`; no complete `resilience_report.json` was produced.
- **Blocker:** Upstream NIM timeout/rate limiting prevented reliable completion of the live proof in this run. Retry only within the pre-registered 300-call cap after service availability allows a complete run.


## Attempt 3 — interrupted live retry after Retry-After change
- Recorded at (Asia/Calcutta host clock): 2026-11-10T20:36:15.4520955+05:30
- Code under test: dbfb6d890bb8cb2f986446bbd33f8d6762fba33f
- Command: .venv\Scripts\python.exe -u scripts\verify_task7_resilience.py with ENABLE_BACKEND_KEEPALIVE=false.
- Outcome: offline 20-request loop lag reported 44.304 ms; /health returned 200. One live orchestration request then logged an SDK RateLimitError retry. The user instructed stop; run was interrupted before orchestration or SSE completed.
- HTTP response body/headers were not captured on this attempt, so the exact upstream 429 status/body/Retry-After value cannot be established from this transcript.
- Raw attempt files were moved without editing to erification/task_7/failed_attempts/ and are failed-attempt evidence only.
- Source file timestamps and sizes at preservation:

Name                                     LastWrite                         Length
----                                     ---------                         ------
verification\task_7\gates_output.txt     2026-11-10T20:30:09.0343444+05:30   9791
verification\task_7\live_run_current.txt 2026-11-10T20:30:34.1587543+05:30    192



## Attempt 4 — local real HTTP request with initial wire-capture instrumentation
- Recorded: 2026-11-10 20:48–20:50 Asia/Calcutta; code at commit `9ea31c9`.
- Started the real FastAPI server with the opt-in capture path; `/health` returned 200 and reported NVIDIA and Pinecone configured.
- Sent one real `POST /orchestrate` request with request ID `task7-orchestrate-real-001`. The backend logger reported an NVIDIA `RateLimitError` retry (3.49s); a later retry logged `TimeoutError` (4.22s). The client run was interrupted before it received a response. No SSE request was made.
- Pinecone index description, embedding, and query requests were captured; all recorded responses were 200. The initial capture hook missed NVIDIA's shared client's default transport, so this attempt does not establish the NVIDIA HTTP status/body/Retry-After header or a complete request trace.
- Raw captured traffic and server log are preserved under `verification/task_7/failed_attempts/live_real_20261110/`. This is failed, incomplete evidence only.
- Root cause in capture instrumentation was patched in commit `54d3bcc`; default-client transport construction was locally verified to instantiate `WireSpyAsyncHTTPTransport`. No subsequent live retry has been run yet.

## Attempt 5 — real HTTP orchestration with corrected wire capture
- Recorded: 2026-11-10 20:55 Asia/Calcutta; server started from committed code `a2991fb`.
- `/health` returned 200. The real `/orchestrate` request reached Open-Meteo (200), Pinecone (embedding/index/query traffic; recorded in the raw capture), and NVIDIA NIM.
- NIM returned 200 twice and then HTTP 429 twice (`Too Many Requests`). The 429 responses had no `Retry-After` header. Captured request ID: `task7-orchestrate-real-001`; no API key values were captured.
- The then-current client retried the two 429s after 3.15s and 7.07s. The client was interrupted and server stopped before an HTTP response or SSE check. These automatic short retries were the defect fixed in commit `2c321c2`: a 429 without a valid `Retry-After` now fails fast; a focused test asserts exactly one upstream request.
- Raw transport evidence and server logs are in `verification/task_7/failed_attempts/live_real_20261110_attempt2/`. This is failed/incomplete evidence; it does not pass Task 7.
