# Task 4 Failures & Root-Cause Log (`R4`)

## Attempt 1 (`task-1221`, committed code `b683e4a`)
- **Outcome:** Live run `task4-live-odisha-cyclone` exited with `status="insufficient_data"` (`llm_calls=8`, `tool_calls=28`, `tokens=28076`, `wall_ms=167526.53`, `citations=0`, `quantities=0`, `abstentions=3`).
- **Observed Root Causes (`verification/task_4/live_traces.json` & single-step replay `task-1251`):**
  1. **Cause A (`geocode_region` returned `no_data` for `"Odisha, India"`):** Open-Meteo `/v1/search?name=Odisha, India` returned empty results because of the comma-separated suffix, whereas splitting `"place, country"` and filtering by country resolves it cleanly without hardcoding.
  2. **Cause B (Risk tool loop + token exhaustion):** On Steps 1–3, Risk succeeded on `search_precedents`, `get_historical_stats`, and `get_weather`. In `task-1251` (Step 4 replay), `reasoning_content` explicitly concluded `"Let me prepare the report."`, yet under free `tool_choice="required"` across repeated unconstrained tool schemas (`$defs/$ref` + full previous tool list + repeated `tool_results`), NIM's decoder re-emitted the previous 4 domain tool calls (`search_precedents`, `get_historical_stats`, `geocode_region`, `get_weather`) on Steps 4 and 5 instead of invoking `submit_risk_report`. That hit Risk's `16000`-token cap.
  3. **Cause C (Unbounded Coordinator re-delegation):** After Risk abstained on token budget, Coordinator re-invoked `run_risk_agent` a second time, repeating the token burn and exhausting the `150.0s` wall-clock cap.
- **Hung Command Incident (`task-1235`):** Diagnostic replay via PowerShell `@' ... '@ | python -` stalled with empty stdout for ~28 minutes waiting on piped stdin; killed manually. Subsequent diagnostics must write files under `scratch/` and pass explicit timeouts.
- **Structural Fixes Planned Before Attempt 2:**
  - Per-agent tool-round cap (`3`) + forced named `tool_choice` to `submit_*_report` once required tools have succeeded or round cap/duplicate trigger is hit.
  - Idempotent per-agent tool deduplication (`already_provided` marker; repeated identical calls trigger forced submit).
  - Compact tool result summaries returned to the model (hard char cap; full payloads kept in traces/state).
  - `parallel_tool_calls=True` enabled cleanly in client calls.
  - Shared `run_budget` + no re-delegation to abstained/out-of-budget agents (only 1 retry allowed, only if budget remains and only on verifier rejection).
  - `"place, country"` parsing & country filtering in `geocode_region` + coordinate-skip rule when lat/lon supplied.

## Attempt 2 (`task-1363`, committed code `b1bc1b5`, interrupted by server restart)
- **Outcome:** `scripts/run_gates.py` passed (`35 passed, FINAL_EXIT=0`) and `capture_scripted_gates()` completed cleanly, then the live suite started `[LIVE RUN START] flood_strong_precedent_run1 ...` and had not emitted an output line before the host server restarted (`[Notice] All your subagents and background tasks have been stopped due to server restart`). Zero completed scenario summaries existed on disk under `verification/task_4/raw_traces/`.
- **Fix Before Attempt 3:** Added live timestamped `on_event` progress logging + per-scenario `--only` / incremental checkpoint saving in `scripts/verify_task4_live.py` with explicit per-scenario timeout.

## Attempt 3 (`task-1378`, committed code `fc99016`, partial pass revealing Logistics citation & timeout bugs)
- **Outcome:** Single live test `flood_strong_precedent_run1` reached `status="partial"` (`risk="high"`, `citations=5`, `quantities=8`, `llm_calls=7`, `tool_calls=8`, `tokens=20589`, `wall_ms=198647.91`), proving Risk verified in 2 turns and Coordinator finished cleanly, **but** Logistics failed verification (`cited_record_ids` contained tool `call-*` IDs instead of EM-DAT IDs because `sub_context` only passed `retrieved_count` without `retrieved_record_ids`) and then hit `ToolSpec(timeout_s=95.0)` while Coordinator skipped Evacuation afterward.
- **Fix Before Retry:** Pass `retrieved_record_ids` directly to Logistics & Evacuation; clarify `cited_record_ids` vs `provenance` in `prompts/logistics.md`; instruct Coordinator in `prompts/coordinator.md` to invoke `run_evacuation_agent` even if Logistics abstains; set specialist `max_tokens=3200`, `wall_clock_cap_s=150.0`, and delegation `ToolSpec` `timeout_s=180.0` (`run_deadline=450.0s`).



