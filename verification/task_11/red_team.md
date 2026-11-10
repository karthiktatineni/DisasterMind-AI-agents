# Task 11 Packaging and Security Review

## Completed checks

- Python runtime pinned at 3.14.6; `requirements.lock` resolves 52 packages from `requirements.txt`.
- A fresh `.venv` installed from `requirements.lock`; `scripts/run_gates.py` passed (`59 passed, 1 deselected`, `FINAL_EXIT=0`). This exposed and fixed the missing direct test dependency `tiktoken`.
- `pip-audit -r requirements.lock`: no known vulnerabilities.
- `npm audit --prefix frontend`: 0 vulnerabilities after adding the locked Playwright UI test dependency.
- Current tree and reachable Git history scanned for credential-shaped patterns. The only historic matches hash to the explicitly allowlisted synthetic negative-control strings; the scanner prints no secret values. Scan result: `HISTORY_SECRET_SCAN_PASS`.
- Built frontend static assets contain no provider-key patterns or `DISASTERMIND_API_KEY`.
- Backend regression suite includes auth, CORS, proxy, rate-limit, concurrency, and chunked-body checks; all pass through `run_gates.py`.
- Production frontend CSP excludes `unsafe-eval`, limits connections to same-origin APIs, and includes framing/content-type/referrer protections. Browser smoke test asserts production security headers.
- No pickle serialization usage found in backend/frontend source.
- `docs/REPRODUCIBILITY.md` documents lock installation and local setup.

## Open items

- This is a fresh locked virtual environment in the existing checkout, not a separately cloned Git worktree.
- EM-DAT legal/display review, the public remote history remediation, and Pinecone account quota/billing questions remain tracked in `verification/OPEN_ITEMS.md` and require owner action.
