# Pre-Task-4 Capability Probe Report (`nvidia/nemotron-3-ultra-550b-a55b`)

- **Target Model:** `nvidia/nemotron-3-ultra-550b-a55b`
- **Endpoint:** `https://integrate.api.nvidia.com/v1`
- **Raw Evidence:**
  - [`models_catalog.json`](models_catalog.json)
  - [`probe_report.json`](probe_report.json)
  - [`probe_trials_20.json`](probe_trials_20.json)
  - [`probe_rpm_seed_phase2.json`](probe_rpm_seed_phase2.json)
- **Live Chat Requests Used:** `52` (Phase 1) + `48` (Phase 2) = **`100` calls** (within 400-call Task 4 cap per R12).
- **Pre-Task-4 Gate Decision:** **PASS (GO)** — Native `tool_calls` achieved **`20/20 (100.0%)`** Pydantic schema validity across 20 trials (`>= 95%` required).

## Capability Matrix

| Capability | Measured Result | Evidence / Engineering Implication |
|---|---|---|
| **Catalog Entry (`GET /v1/models`)** | `200 OK`, `84` models, exact match `"nvidia/nemotron-3-ultra-550b-a55b"` (`owned_by="nvidia"`) | Metadata schema omits `context_window` (`{id, object, created, owned_by}` only). |
| **Native `tools` & `tool_choice`** | `auto=True`, `required=True`, named function=`True` | Full native OpenAI-compatible `message.tool_calls` support; zero text/regex parsing needed for tool routing. |
| **`parallel_tool_calls`** | `parallel_tool_calls=True` emitted `2` valid parallel calls (`get_historical_stats`, `get_weather`); `False` emitted `1` call | Supports parallel multi-tool batching in one turn when enabled. |
| **`response_format`** | `json_object`: `200 OK`, valid JSON (`True`); `json_schema` (`strict: true`): `200 OK`, valid against `ProbeBriefSchema` (`True`) | Constrained JSON schema works reliably; include schema instructions in system/developer prompt too so reasoning aligns cleanly with schema constraints. |
| **`seed` Determinism** | `T=0.0, seed=424242` tool pair: identical `reasoning_content` and parsed `(name, args)` (`True`, server `call-*` IDs differ). `T=0.0, seed=424242` JSON pair: identical `content` (`True`) **but different `reasoning_content` (`False`)**. `T=0.7` trio: `seed=424242` (x2) and `seed=999999` (x1) all emitted identical JSON; `system_fingerprint=null`. | Server-side reasoning tokens are **not** guaranteed bit-identical even with identical `seed` and `T=0.0`; Tasks 4–9 use `T=0.0` + strict code/Pydantic verification rather than assuming NIM seed reproducibility. |
| **Context Window** | `max_tokens=1_000_000` returned `200 OK` (clamped by server); API catalog omits context length. Official NVIDIA NIM spec: **262,144 tokens (256K)** default (`1M` max extended architecture limit). | Safe headroom for 4-agent prompts with bounded context budgets. |
| **HTTP `410` Deprecated Model** | `nvidia/llama-3.3-nemotron-super-49b-v1` -> **HTTP `410 Gone`** (`application/problem+json`, EOL `2026-08-26T09:00:00Z`) | Verified live both here and via `scripts/preflight.py --control bad-model-410` (`EXIT 1`). |
| **RPM Limit & `429` Behavior** | Burst of `20` in `3.452 s` -> `20 x 200 OK`, `0 x 429`. Wave of `3` + `45` rapid requests (`3.375 s`) -> **`25 x 200 OK` + `20 x 429 Too Many Requests`** (`application/problem+json`, `{"status": 429, "title": "Too Many Requests"}`, `nvcf-error-source: upstream_status`). **Zero `x-ratelimit-*` or `Retry-After` headers emitted.** | Consistent with NVIDIA Build free-tier ~`25–40 RPM` limit; because NIM omits `Retry-After` headers, Task 4/7/9/10 must enforce a client-side RPM limiter + bounded concurrency (`<= 5`) + exponential backoff with jitter on `429`. |
| **20-Trial Required Tool Benchmark** | **`20/20 (100.0%)`** valid schema-checked tool calls (`lat_min=1621.4 ms`, `p50=3867.42 ms`, `mean=4418.22 ms`, `p95=14871.73 ms`, mean completion `111.8` tokens incl. `58.7` reasoning tokens) | Passes `>= 95%` stop gate cleanly; keep `max_tokens >= 1024` so thinking tokens never truncate `tool_calls`. |

