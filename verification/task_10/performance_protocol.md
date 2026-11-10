# Task 10 Performance and Cost Protocol

## Capture safety

Instrument outbound `httpx` requests with a `_WireSpy` response hook that records one JSON object per completed provider request in `verification/final_checks/network_capture.jsonl`. Capture request correlation ID, provider host/path, status, elapsed milliseconds, retry count, and provider-reported token usage only. Never write prompts, completions, authorization headers, API keys, or raw response bodies. Record only aggregate token fields after parsing in memory.

## Workload matrix

Run the frozen representative DEV workload at orchestration concurrency 1, 5, and 20. For each concurrency level run one cold batch (fresh app lifespan/client pool) and one warm batch (reuse the same lifespan/client pool). Use scenario-specific request IDs so all HTTP, SSE, and wire records can be reconciled. Keep retrieval `top_k <= 5`; do not enumerate or re-embed the production namespace.

Report p50/p95/p99 and maximum end-to-end latency, provider-request latency, token totals, cost using the configured public model price table (record its dated source), 429 count, timeout count, retries, failed/partial briefing count, and sample size. Preserve raw JSONL and a compact aggregate JSON report.

## Acceptance and stop conditions

- Hard cap: 1,200 NIM calls. Planned range: 200–500 calls; count actual provider attempts, including retries.
- Stop on repeated 429s, any unbounded retry, a 500 response, missing request correlation, or cap exhaustion.
- Use the phrase “sub-second p95” only if the measured p95 is strictly below 1,000 ms for the stated matrix cell; otherwise publish the actual latency and remove the claim.
- Compare cold and warm batches separately. Never discard failed requests from latency or reliability denominators.

## Current state

Protocol is pre-registered; no performance calls have been made. Task 7 observed upstream NIM timeout/rate-limit responses. Wait for service availability, then run within the 1,200-call cap.
