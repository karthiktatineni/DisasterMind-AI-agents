# DisasterMind AI — Baseline Verification & Audit Closeout (Task 0)

**Date:** 2026-11-10  
**Branch:** `fix/adversarial-qa`  
**Role:** Principal Engineer & Adversarial QA Lead  
**Evaluation Mode:** Pre-Fix Baseline (Deterministic Fallback / Offline Model State)

---

## 1. Test Suite Baseline

### 1.1 `python -m pytest tests/`
- **Command:** `python -m pytest tests/`
- **Result:** PASS (6 passed in 17.21s)
- **Log Evidence:**
  ```text
  tests\test_nvidia_generation.py ..                                       [ 33%]
  tests\test_orchestration.py ....                                         [100%]
  ============================= 6 passed in 17.21s ==============================
  ```

### 1.2 Bare `pytest` Defect Confirmation
- **Command:** `pytest`
- **Result:** FAIL (`ModuleNotFoundError: No module named 'backend'`)
- **Finding:** Python path lacks working directory resolution during bare invocation; `pyproject.toml` with `pythonpath = ["."]` or package install is required.

---

## 2. Pre-Fix Retrieval Evaluation Baseline (Exact Cosine Simulator on 12,000 Records)

- **Script:** `eval/audit_test_suite.py` (migrated from `verification/audit_test_suite.py`)
- **Evaluated Dataset:** 16,834 curated EM-DAT records, 12,000 embedded (4,096 dimensions)
- **Results:**
  - **Total Test Queries:** 30
  - **Domain-Relevant Queries:** 21
  - **Recall@1:** 0.5714 (12 / 21)
  - **Recall@5:** 0.6190 (13 / 21)
  - **Recall@10:** 0.7143 (15 / 21)
  - **Mean Reciprocal Rank (MRR):** 0.6012
  - **Failing Queries:** 12 total
    - `[Q09] 'Cyclone Idai Mozambique 2019'`: Rank = None (Failed to retrieve Storm/Mozambique in top 10)
    - `[Q12] 'erthquak pakstan 2005'`: Rank = None (Typo query failed vector search)
    - `[Q13] 'huge ocean water surge and wind destroying houses'`: Rank = None (Descriptive query failed without keyword match)
    - `[Q18] 'stock market crash and bank bankruptcy in New York'`: False Positive (Out-of-domain query retrieved disaster records above zero-floor)
    - `[Q20] 'presidential election voter turnout and protest riot'`: False Positive (Out-of-domain query retrieved disaster records)

---

## 3. Pre-Fix Grounding & Hallucination Baseline (Phase 6 Suites)

| Test Suite | Metric / Measure | Pre-Fix Baseline | Primary Finding & Failure Cause |
|---|---|---|---|
| **6.a Answerable-from-data** | Pass Rate | 8 / 8 (100%) | Simple matches retrieve historical incidents |
| **6.b Unanswerable (Missing)** | Pass Rate | 8 / 8 (100%) | Emits missing precedent notification |
| **6.c False Premise** | Pass Rate | **0 / 8 (0.0%)** | **CRITICAL:** Blindly accepted all 8 physically impossible scenarios (e.g. Category 6 hurricane in Paris, M11.5 earthquake) |
| **6.d Numeric Fabrication** | Ungrounded Number Rate | **461 / 767 (60.1%)** | **CRITICAL:** 60.1% of operational numbers (ambulances, food kits, headcounts) generated from arbitrary mathematical divisors in `fallbacks.py` |
| **6.e Citation Validity** | Citation Validity Rate | 80 / 80 (100.0%) | Historical citations match format, but event names frequently render `None (Year)` |
| **6.f Prompt Injection** | Injection Resistance | 8 / 8 (100%) | Fallback heuristics ignore user notes |
| **6.g Conflicting Sources** | Conflict Surfaced Rate | **0 / 8 (0.0%)** | **CRITICAL:** Conflicting historical reports silently averaged or first record chosen |
| **6.h Failure Injection** | Resiliency Score | **6 / 8 (75.0%)** | **CRITICAL:** 2 unhandled 500 crashes (`RuntimeError: DB connection dropped` in `orchestrator.run()`) |
| **6.i Consistency** | Variance across 10 runs | 0.00 variance, 1 unique output | Artificial pass: 100% deterministic fallback path executed because live LLM returned HTTP 410 |
| **6.j Error Cascading** | Cascading Fault Caught | **0 / 8 (0.0%)** | Contradictory upstream risk score blindly propagated through validation to final plan |

---

## 4. Pre-Fix Latency Benchmarks

Measured on local hardware running exact cosine vector retrieval over 12,000 embedded records, end-to-end fallback orchestration, and external enrichment APIs.

| Component / Path | p50 (ms) | p95 (ms) | p99 (ms) | Mean (ms) | Min (ms) | Max (ms) | Notes |
|---|---|---|---|---|---|---|---|
| **Vector Search (12k records)** | **9.14** | 10.94 | 21.39 | 9.70 | 7.55 | 30.83 | Exact in-memory numpy matrix dot product (simulated pgvector exact scan) |
| **Agent Pipeline E2E (Fallback)** | **122.10** | 571.91 | 7,154.98 | 556.60 | 115.99 | 8,800.75 | Pure Python deterministic fallbacks; tail latency driven by file I/O |
| **External Data Enrichment** | **11,951.50** | 12,090.96 | 12,111.09 | 8,068.76 | 2,062.90 | 12,116.12 | Serial external HTTP calls to Open-Meteo, Nominatim, Overpass (dominates e2e latency) |

---

## 5. Secret Scan & Repository Hygiene Report

### 5.1 Working Tree & Git History Scan
- **Scanner:** Automated AST and regex scanner across git commit history (`git log -p --all`) and all working tree files.
- **Git History Findings:** **Zero real production secrets in git history.** Only dummy string `"test-key"` in `tests/test_nvidia_generation.py`.
- **Working Tree Findings:**
  1. `.env`: Contains real `NVIDIA_API_KEY` (nvapi-...), `SUPABASE_SERVICE_ROLE_KEY` (JWT), `SUPABASE_DB_PASSWORD`, and `SUPABASE_URL`.
  2. `.cursor/mcp.json`: Contains Supabase project reference `yfxlavxflmvvxwzlwiwa`.
  3. `upload_report.md` & `supabase_connection_report.md`: Contain project IDs and operational metadata.
- **Git-Ignore Verification:**
  - `.env` is confirmed git-ignored (`.gitignore:1`) and untracked.
  - `logs/` is confirmed git-ignored and untracked.
  - `.cursor/` is confirmed git-ignored and untracked.
  - Stale reports archived to `docs/archive/` have been cleansed of secret values.
- **Keys Requiring Rotation:**
  - `NVIDIA_API_KEY`
  - `SUPABASE_SERVICE_ROLE_KEY`
  - `SUPABASE_DB_PASSWORD`

### 5.2 Large File & Object Storage Analysis
- **`git count-objects -vH` output:**
  ```text
  count: 1178
  size: 2.84 GiB
  in-pack: 0
  packs: 0
  size-pack: 0 bytes
  ```
- **Identification:**
  - `embeddings/embedded_disasters_output.json` (1,038,842,071 bytes ~ 1.03 GB) exists on disk in working tree.
  - **Git History Check:** Verified via `git rev-list --objects --all` that this 1.03 GB file was **never committed** to git history. The 2.84 GiB loose objects in `.git/objects` represent dangling objects from previous uncommitted local `git add` operations.
- **Proposed Plan:**
  1. Keep `embeddings/*.json` in `.gitignore` (already active).
  2. Never track raw large JSON embeddings in git.
  3. Clean up dangling local objects via `git gc --prune=now` only if approved by user, without rewriting commit history.

---

## 6. Frontend Audit Addendum

Full audit of `frontend/app/page.tsx` (lines 331–846) and `frontend/app/api/`:

| File | Line(s) | Finding & Vulnerability Detail | Invariant / Rule Affected |
|---|---|---|---|
| `frontend/app/page.tsx` | 259–264 | **Fake Animation Loop:** Runs artificial `setTimeout(resolve, 130)` per agent (1.69s total) before sending the HTTP request. Agents display as "complete" before the backend runs. | **I8** |
| `frontend/app/page.tsx` | 61–75 | **13 Sequential Agent Stages:** Hardcoded UI list of 13 agents, 9 of which never invoke an LLM. | Target Architecture |
| `frontend/app/page.tsx` | 391–421 | **Unvalidated Inputs:** Free text region without coordinate bounds; no Category dropdown bounds for cyclones; no magnitude scale for earthquakes. | **I5** |
| `frontend/app/page.tsx` | 510–543 | **Misleading Metrics:** Conflates predicted affected population with evacuation need (`asNumber(predicted_affected, evacuation_need)`); validation status shows "Running" or "Passed" with no human review flag. | **I4, I9** |
| `frontend/app/page.tsx` | 608–632 | **Unlabeled Historical Data:** Historical disasters rendered without explicitly distinguishing them from the hypothetical active disaster. | **I9** |
| `frontend/app/page.tsx` | 706–724 | **Missing Request Cancellation:** Rapid resubmissions do not abort previous inflight fetch requests (no `AbortController`). | UI UX / Concurrency |
| `frontend/lib/api.ts` | 91–92, 122 | **Bypass of Data Enrichment:** When `NEXT_PUBLIC_API_BASE_URL` is set, calls `${API_BASE_URL}/orchestrate` directly instead of `/api/orchestrate` -> `/orchestrate-intake`, bypassing Open-Meteo and OSM enrichment. | Data Integrity |
| `frontend/app/api/orchestrate/route.ts` | 6–43 | **No Streaming / SSE Support:** Uses simple POST buffering with 502 error masking instead of SSE event streaming. | **I8** |
| `frontend/app/api/agent-answer/route.ts` | 6–42 | **Unsanitized Proxy:** Forwards unvalidated user question directly to backend; returns raw text on failure without error schema. | API Hygiene |

---

## 7. Primary Source Verification Evidence

### 7.1 NVIDIA Live Models & 410 Verification
Live API query against `https://integrate.api.nvidia.com/v1`:
- **Live Endpoint Response:**
  - `GET /v1/models` Status: **200 OK** (84 active models returned).
  - Configured Chat Model `nvidia/llama-3.3-nemotron-super-49b-v1`: **NOT in available models**.
  - Configured Embedding Model `nvidia/nv-embed-v1`: **NOT in available models**.
- **Chat Completion Test (`nvidia/llama-3.3-nemotron-super-49b-v1`):**
  - **Status:** `HTTP 410 Gone`
  - **Body:**
    ```json
    {
      "type": "about:blank",
      "title": "Gone",
      "status": 410,
      "detail": "The model 'nvidia/llama-3.3-nemotron-super-49b-v1' has reached its end of life on 2026-08-26T09:00:00Z and is no longer available."
    }
    ```
- **Embedding Test (`nvidia/nv-embed-v1`):**
  - **Status:** `HTTP 410 Gone`
  - **Body:**
    ```json
    {
      "type": "about:blank",
      "title": "Gone",
      "status": 410,
      "detail": "The model 'nvidia/nv-embed-v1' has reached its end of life on 2026-08-25T09:00:00Z and is no longer available."
    }
    ```

### 7.2 Frontend `npm audit` Vulnerability Details
- **Installed Next.js Version:** `16.2.7`
- **Total Audit Findings:** 7 vulnerabilities (1 Critical, 5 High, 1 Moderate) across Next.js and dependencies.
- **Direct Next.js Severity:** **CRITICAL** (Vulnerable Range: `9.3.4-canary.0 - 16.4.0`)
- **Key Advisories:**
  - `GHSA-p293-qw3h-jr36` (Critical): Unauthenticated Remote Code Execution on Windows-hosted servers (`>=16.0.0 <16.3.3`)
  - `GHSA-2xp9-vwfh-vxw4` (Critical): Unauthenticated Remote Code Execution in Image Optimization API when AVIF files are used (`>=16.0.0 <16.3.3`)
  - `GHSA-vcvr-r3jv-pc5j` (Critical): Remote Code Execution in `next/og` ImageResponse (`>=16.2.0 <16.3.6`)
  - `GHSA-2464-fhr4-w9xw` (Critical): Unauthenticated Remote Code Execution in Image Optimization API when SVG files are used (`>=16.0.0 <=16.4.0`)
  - `GHSA-673q-mjm6-r5xr` (Critical): Unauthenticated Remote Code Execution in `use cache` functions (`>=16.0.0 <=16.4.0`)
  - `GHSA-6gpp-xcg3-4w24` (High): Middleware / Proxy bypass in App Router (`>=16.0.0 <16.2.11`)
  - `GHSA-m99w-x7hq-7vfj` (High): Denial of Service in App Router using Server Actions (`>=16.0.0 <16.2.11`)
- **Resolution:** Upgrade `next` in `frontend/package.json` to `>=16.4.6`.

### 7.3 EM-DAT Terms of Use & Licensing Constraints
- **Primary Source:** Verified against EM-DAT CRED / UCLouvain documentation (`https://www.emdat.be` and `data/public_emdat_custom_request_2026-06-09_9221c56d-9867-40b3-9b51-cef53eda41e3.xlsx` sheet `EM-DAT Info`).
- **Permitted Use:** Free non-commercial access strictly for academic research, teaching, or non-profit humanitarian/informational purposes.
- **Commercial Restrictions:** Any commercial use or use by corporate entities requires a separate paid Database License Agreement from UCLouvain/CRED.
- **Redistribution Restrictions:** Strict prohibition against copying, reproducing, or redistributing substantial portions of the raw database or creating substitute databases.
- **Attribution Policy:** Mandatory citation: *"EM-DAT, CRED / UCLouvain, Brussels, Belgium - www.emdat.be"*.
- **Documentation Action:** README and schema documentation must state non-commercial research licensing terms clearly.

---

## 8. Sealed Holdout Dataset (Rule 6)

The sealed holdout evaluation suite has been authored, locked, and committed to `eval/holdout_sealed.json`.
- **File:** `eval/holdout_sealed.json`
- **Total Retrieval Queries:** 25 (Exact historical, out-of-domain, typos, multilingual, descriptive without keywords, unrepresented combinations)
- **Total Scenarios:** 25 (Standard multi-hazard, high severity, false premises, data gaps, prompt injections, conflicting reports, shelter/resource starvation)
- **SHA-256 Hash:**
  ```text
  bbb454c7a4067f45e0253de6894ceb562ad9b922ee611318373e0b6141f1ec6f
  ```
- **Sealing Commitment:** Under Anti-Fake-Success Rule 6, this holdout dataset will **NOT** be inspected, run, or used for parameter tuning during Tasks 1–11. It will be run exactly once for final scoring in Task 12.
