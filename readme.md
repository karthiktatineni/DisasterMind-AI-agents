# DisasterMind AI

**Evidence-grounded, 4-agent emergency planning and historical disaster briefing platform.**

DisasterMind AI assists emergency coordinators with scenario-based risk, logistics, and evacuation briefings grounded in calibrated EM-DAT historical retrieval (Pinecone), deterministic planning standards (WHO/WEDC & Sphere), live Open-Meteo geocoding/weather lookups, and strict server-side verification.

> **Operational Safety Notice:** Outputs are scenario-planning decision support only. The system is not an autonomous command platform and does not replace local emergency authorities, live hazard warnings, or real-time shelter/inventory feeds. Every briefing enforces `requires_human_approval = true` in code.

---

## System Architecture

```
Next.js Frontend (frontend/)
  ├── Client UI (app/page.tsx) ──POST──▶ /api/orchestrate/stream (Same-origin SSE)
  └── Server Route Handlers (app/api/*)
        └── Injects server-only X-DisasterMind-API-Key (never NEXT_PUBLIC_*)
              │
              ▼
FastAPI Backend (backend/app/main.py)
  ├── Pure-ASGI Security Stack (backend/app/security.py)
  │     RequestContext -> CORS -> SecurityHeaders -> SizeLimit (64KB) -> RateLimit -> ConcurrencyCap
  ├── Lifespan Shared Clients & Startup Guard (snapshot_guard.py)
  │     Verifies config/index_manifest.json <-> Settings <-> live Pinecone index
  └── 4-Agent Coordinator Orchestrator (backend/app/orchestration/orchestrator.py)
        ├── Coordinator ──▶ Risk ──▶ Logistics ──▶ Evacuation ──▶ Final Synthesis
        ├── NVIDIA NIM Chat LLM (nvidia/nemotron-3-ultra-550b-a55b, native tool calling)
        ├── InAppVerifier (numeric provenance, narrative check, citation check, approval guard)
        └── Domain Tools (backend/app/tools/domain_tools.py)
              ├── search_precedents ──▶ Pinecone (llama-text-embed-v2, 1024-dim, top_k <= 5)
              ├── get_historical_stats ──▶ Optional local Parquet snapshot (n >= 5 guardrail)
              ├── geocode_region & get_weather ──▶ Open-Meteo APIs
              └── compute_logistics_requirements ──▶ config/planning_standards.json
```

---

## Component Breakdown

### 1. Four-Agent Native Tool-Calling Orchestration (`backend/app/orchestration/`)

DisasterMind AI uses **four LLM agents** coordinated strictly through OpenAI-compatible native `tools` and `tool_choice` (`seed=42`, `temperature=0.0`, `max_tokens=3200`):

| Agent | Prompt | Allowed Tools | Step / Round / Budget Limits | Role |
|---|---|---|---|---|
| **Coordinator** | `prompts/coordinator.md` | `search_precedents`, `geocode_region`, `run_risk_agent`, `run_logistics_agent`, `run_evacuation_agent` | 8 steps, 5 tool rounds, 36k tokens, 420s wall cap | Resolves geography/precedents, delegates to specialists, synthesizes final briefing |
| **Risk** | `prompts/risk.md` | `search_precedents`, `get_historical_stats`, `geocode_region`, `get_weather` | 6 steps, 3 tool rounds, 24k tokens, 150s wall cap | Evaluates scenario physical premise (`premise_valid`), hazard level, and meteorological context |
| **Logistics** | `prompts/logistics.md` | `search_precedents`, `get_historical_stats`, `compute_logistics_requirements` | 6 steps, 3 tool rounds, 24k tokens, 150s wall cap | Calculates daily water requirement (`15.0 L/person/day`) and shelter demand against user-supplied inventory |
| **Evacuation** | `prompts/evacuation.md` | `search_precedents`, `get_historical_stats`, `get_weather` | 6 steps, 3 tool rounds, 24k tokens, 150s wall cap | Recommends evacuation posture (`no_evacuation` to `mandatory_evacuation` or `insufficient_data`), trigger criteria, and human sign-off |

Agent outputs use strict Pydantic models (`extra="forbid"`) in `backend/app/orchestration/models.py`. On upstream tool errors, missing precedents, or LLM unavailability, agents abstain cleanly and return `insufficient_data` or `partial` without inventing substitute numbers.

### 2. Grounding, Verification & Injection Controls

- **Zero Heuristic Fallbacks (`verifier.py`):** Every numeric claim across structured fields and narrative prose is checked against an allowed-number set built from user inputs, verified tool call outputs (`call_id`), retrieved EM-DAT record fields, and `config/planning_standards.json` (WHO/WEDC Technical Note No. 9 / Sphere `15.0` L/person/day water standard). Unsourced quantities fail verification.
- **Calibrated Retrieval Threshold:** Pinecone semantic search (`search_precedents`) enforces the calibrated similarity floor (`SIMILARITY_THRESHOLD=0.3112` in `config/index_manifest.json`) over index `disastermind`, namespace `v2-3d35e5131d79a79d-llama-text-embed-v2`, using Pinecone-hosted `llama-text-embed-v2` (`1024`-dim).
- **Allow-Listed Metadata Only:** Raw narrative `search_text` is never passed to LLM prompts or API responses; only structured citation fields (`record_id`, `event_name`, `disaster_type`, `country`, `region`, `start_year`, `total_deaths`, `total_affected`, `total_damage_usd_thousands`, `similarity`) are exposed.
- **Prompt Injection Defense (`prompt_guard.py`):** User `notes` and `question` inputs are capped at `500` characters, control and bidi-formatting characters are stripped, breakout tags (`<system>`, `<user_notes>`, `<|im_start|>`, etc.) are neutralized, and untrusted text is wrapped inside `<user_notes>` / `<untrusted_retrieved_metadata>` tags.
- **Mandatory Human Sign-Off:** `EvacuationSummary.requires_human_approval` and `Briefing.human_review.required` are forced `True` in orchestration code regardless of model output, alongside scenario classification (`historical_precedent` vs `hypothetical_scenario`) and operational disclaimers.

### 3. Resilience, Security & Runtime Controls (`backend/app/`)

- **Pure-ASGI Middleware (`security.py`, `main.py`):**
  - `RequestContextMiddleware`: Attaches and logs `X-Request-ID` across responses, logs, and SSE streams.
  - `CORSMiddleware`: Restricts origins to `ALLOWED_ORIGINS` (`GET`, `POST`, explicit headers).
  - `SecurityHeadersMiddleware`: Sets defensive response headers on API replies.
  - `RequestSizeLimitMiddleware`: Enforces `MAX_REQUEST_BYTES` (`65536`) on both `Content-Length` and streamed request chunks (returning `413` cleanly).
  - `InMemoryRateLimitMiddleware`: Per-IP sliding window (`RATE_LIMIT_REQUESTS=60` / `60s`), honoring `X-Forwarded-For` only from configured `TRUSTED_PROXIES`.
  - `ConcurrencyCapMiddleware`: Caps concurrent orchestrations (`MAX_CONCURRENT_ORCHESTRATIONS=2` by default, returning `429` with `Retry-After`).
- **Auth & Error Envelopes:** Every non-health endpoint validates `X-DisasterMind-API-Key` when `APP_ENV=production` or `DISASTERMIND_API_KEY` is configured. Unhandled exceptions return structured JSON envelopes with `request_id` (status `503` instead of raw `500`s or stack traces).
- **Shared Clients & Bounded 429 Handling (`main.py`, `llm_client.py`):** A single shared `httpx.AsyncClient` and `AsyncOpenAI` client are initialized and closed cleanly in FastAPI `lifespan`. Repeated unhinted NVIDIA NIM `429` responses use a single bounded `30s` retry with an `agent_retrying` SSE notification, followed by a `60s` cooldown to prevent cascading upstream retries.
- **Startup Integrity (`snapshot_guard.py`):** `verify_runtime_alignment` checks `config/index_manifest.json` (`16,833` records, `1024` dims, `llama-text-embed-v2`) against active config and live Pinecone index stats when running in production. If the optional local Parquet snapshot (`data/disastermind_snapshot.parquet`) is omitted on deployed hosts, `get_historical_stats` safely degrades without blocking startup.

### 4. Truthful SSE Frontend (`frontend/`)

- Built with **Next.js 16.4.1**, **React 19.2.7**, **Tailwind CSS 4.3.0**, and **TypeScript 6.0.3**.
- **Server-Side BFF Proxy (`frontend/app/api/`):** Browser requests never contact FastAPI directly or read `DISASTERMIND_API_KEY`. Route Handlers forward requests to `BACKEND_API_BASE_URL` with server-side `X-DisasterMind-API-Key` and pass structured errors through cleanly.
- **Real-Time SSE Progress (`app/page.tsx`):** Streams live typed events (`run_started`, `agent_started`, `agent_retrying`, `tool_called`, `tool_result`, `agent_verified`, `agent_rejected`, `agent_abstained`, `briefing_ready`, `run_failed`) with `AbortController` cancellation on resubmit/cancel.
- **Strict UI Grounding:** Displays `Not provided` / `Not calculable` when inventory or capacity inputs are absent, enforces 500-character limits on scenario notes, shows official authorization banners and disclaimers, and hardens headers in `frontend/next.config.mjs` (`X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, `connect-src 'self'` CSP).

---

## API Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | Public | Liveness, config readiness (`nvidia_configured`, `pinecone_configured`, `api_key_required`), and concurrency cap |
| `GET` | `/keepalive` | Public | Keepalive loop status (`ENABLE_BACKEND_KEEPALIVE`, `BACKEND_KEEPALIVE_URL`) |
| `POST` | `/generate` | API Key | Direct NVIDIA NIM chat completion endpoint |
| `POST` | `/orchestrate` | API Key | Synchronous orchestration accepting `DisasterScenario` |
| `POST` | `/orchestrate-intake` | API Key | Synchronous orchestration accepting UI `ScenarioIntakeRequest` |
| `POST` | `/orchestrate/stream` | API Key | Server-Sent Events (`text/event-stream`) 4-agent orchestration stream |
| `POST` | `/agent-answer` | API Key | Summary Q&A wrapper returning `AgentQuestionResponse` |

---

## Repository Layout

```
.
├── .env.example                 # Backend environment variable template
├── .python-version              # Pinned Python 3.14.6
├── pyproject.toml               # Pytest configuration (offline default, -m 'not live')
├── requirements.txt             # Top-level backend dependencies (used by hosts)
├── requirements.lock            # Resolved Python dependency lockfile
├── backend/
│   ├── requirements.txt         # Mirror backend dependencies
│   └── app/
│       ├── main.py              # FastAPI entrypoint, lifespan, routes, middleware stack
│       ├── config.py            # Environment-driven Settings dataclass
│       ├── schemas.py           # Request/response validation & 500-char caps
│       ├── security.py          # Auth, rate limit, size limit, concurrency, error envelope
│       ├── orchestration/       # 4-agent loop, Pydantic models, verifier, prompt_guard
│       ├── prompts/             # System prompts for coordinator, risk, logistics, evacuation
│       ├── services/            # NIM clients, Pinecone retrieval, Open-Meteo, keepalive, guard
│       └── tools/               # Domain tool schemas & execution handlers
├── config/
│   ├── index_manifest.json      # Committed Pinecone index/model/threshold manifest
│   ├── planning_standards.json  # Verified WHO/WEDC 15 L/person/day water standard
│   └── secret_allowlist.json    # Synthetic test-token hashes allowlisted in scripts/run_gates.py
├── frontend/
│   ├── .env.local.example       # Frontend BFF template
│   ├── next.config.mjs          # CSP & security headers
│   ├── package.json             # Next.js 16.4.1 app dependencies
│   ├── app/
│   │   ├── page.tsx             # SSE briefing console & grounded visualizations
│   │   ├── layout.tsx
│   │   ├── globals.css
│   │   └── api/                 # Server-side proxy routes (/health, /generate, /orchestrate*)
│   ├── lib/api.ts               # SSE stream parser & client fetch helper
│   └── tests/task8-ui.test.mjs  # Playwright browser UI tests
├── tests/                       # Offline pytest verification suite (62 tests)
├── scripts/                     # Gate runner (run_gates.py), ingestion & live check scripts
├── eval/                        # Calibrated DEV evaluation sets, checker & sealed holdout
├── embeddings/                  # Offline text builder & Pinecone embedding scripts
├── docs/                        # Schema export, EM-DAT license review, reproducibility & archives
└── verification/                # Task 0-11 ledger, open items, audit logs & probe reports
```

> Sensitive local files and build artifacts (`.env`, `.env.*`, `frontend/.env*.local`, `data/`, `eval/.cache/`, `embeddings/*.json`, `frontend/AGENTS.md`, `verification/**/raw_traces/`) are excluded via `.gitignore`.

---

## Local Setup & Production Hosting

### 1. Local Development

**Backend (Python 3.14.6):**

```powershell
uv venv .venv --python 3.14.6
uv pip install --python .venv\Scripts\python.exe -r requirements.lock
Copy-Item .env.example .env
# Fill PINECONE_API_KEY and NVIDIA_API_KEY in .env
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

**Frontend (Node 24+ / Next.js):**

```powershell
cd frontend
Copy-Item .env.local.example .env.local
# Set BACKEND_API_BASE_URL=http://127.0.0.1:8000 (and matching DISASTERMIND_API_KEY if enabled)
npm ci
npm run dev
```

### 2. Hosting Deployment (Render + Vercel)

#### Backend Service (Render / Docker / Railway)
- **Root Directory:** `.`
- **Python Version:** `3.14.6` (`.python-version`)
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT --workers 1` *(keep `--workers 1` so `InMemoryRateLimitMiddleware`, `ConcurrencyCapMiddleware`, and NIM cooldown state remain consistent per instance)*
- **Required Environment Variables:**
  - `APP_ENV=production`
  - `PINECONE_API_KEY=<your-pinecone-key>`
  - `PINECONE_INDEX_NAME=disastermind`
  - `PINECONE_NAMESPACE=v2-3d35e5131d79a79d-llama-text-embed-v2`
  - `EMBEDDING_PROVIDER=pinecone`
  - `EMBEDDING_MODEL=llama-text-embed-v2`
  - `EMBEDDING_DIMENSION=1024`
  - `SIMILARITY_THRESHOLD=0.3112`
  - `NVIDIA_API_KEY=<your-nvidia-nim-key>`
  - `NVIDIA_MODEL=nvidia/nemotron-3-ultra-550b-a55b`
  - `DISASTERMIND_API_KEY=<strong-random-shared-secret>`
  - `ALLOWED_ORIGINS=https://<your-vercel-domain>.vercel.app`
  - `TRUSTED_PROXIES=<proxy-ip-or-cidr-if-applicable>`

#### Frontend App (Vercel)
- **Root Directory:** `frontend`
- **Build Command:** `npm run build`
- **Server-Side Environment Variables (do NOT prefix with `NEXT_PUBLIC_`):**
  - `BACKEND_API_BASE_URL=https://<your-backend-host>`
  - `DISASTERMIND_API_KEY=<same-strong-random-shared-secret>`

---

## Verification & Open Operational Notices

- **Offline Verification Suite:** Automated tests cover data pipeline quarantining, historical stats (`n >= 5`) guardrails, Pinecone score ordering, strict numeric grounding & verifier mutation checks, 10 prompt-injection variants, ASGI middleware / 20-request async loop lag / 10-mode failure matrix (`62` offline backend tests), and Playwright UI checks (`frontend/tests/task8-ui.test.mjs`, `2/2` against production build).
- **External Dependencies & Terms:**
  - **EM-DAT:** Historical records in Pinecone derive from the EM-DAT International Disasters Database (UCLouvain / CRED). Confirm EM-DAT license compliance for any public deployment.
  - **NVIDIA API Catalog:** Trial API keys are restricted under NVIDIA's evaluation terms; ensure an appropriate NVIDIA NIM tier/subscription and rate headroom before high-concurrency production traffic.
  - **Pinecone Starter Plan:** Monitor monthly egress (`1 GB` limit on Starter tier) and keep retrieval `top_k <= 5`.
