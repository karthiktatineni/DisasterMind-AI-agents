# Task 9 DEV Live Evaluation Protocol

## Freeze and scope

`eval/scenarios_dev.json` is a frozen copy of the 20 open development scenarios in `eval/dev_set.json`. It excludes `eval/holdout_sealed.json`. The independent checker is `eval/independent_checker.py`; verify its digest against `verification/task_9/checker.sha256` before scoring.

## Runs

1. Run all 20 frozen scenarios three times each (60 runs total), preserving raw request/response JSON, request IDs, model IDs, and timestamps under `verification/task_9/raw_traces/`.
2. Run the pre-registered 8-scenario grounded subset three times (24 runs), and separately run three representative scenarios ten times each (30 runs) for determinism. Do not tune prompts or thresholds during a batch.
3. Count API calls before and after each batch. Hard stop at 1,500 NIM calls, repeated 429 responses, or any production namespace read above `top_k=5`.
4. Score outputs with the frozen checker. Report completion/abstention, citation record-ID integrity, premise validity, mandatory human approval, unsupported numeric assertions, repeat agreement, latency, and token/call usage. Report each metric with its numerator, denominator, and raw trace locator.
5. Any checker failure or unexpected claim blocks PASS. Record the exact failure and root cause before repairs; after a repair, rerun the complete pre-registered batch on the new commit.

## Current execution state

The scenario set and independent checker are prepared; its 13 mutation cases all reject as expected. Live scoring has not started. The upstream NIM service returned timeout/rate-limit responses during Task 7, so do not spend the Task 9 budget until service availability returns. Planned spend: approximately 250–600 calls, hard cap 1,500.
