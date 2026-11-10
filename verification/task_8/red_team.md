# Task 8 Frontend Grounding Red-Team Notes

## Code-path review

- Agent progress is driven by backend SSE event names and agent IDs; no timer or fabricated completion state remains.
- The visible pipeline contains only the canonical coordinator, risk, logistics, and evacuation agents.
- Scenario notes are capped at 500 characters in the UI and truncated to the same limit at the API call boundary.
- Cancel aborts the active fetch; stream errors and `run_failed` events surface as an error and do not create a briefing.
- The UI renders the briefing label, premise validity, citations, gaps, conflicts, abstentions, disclaimers, and human review fields from backend data. Missing numbers and claims display as unavailable/not stated rather than zero or estimated values.
- API key is attached only by the server route. The stream route forwards upstream error status/body and disables intermediary buffering.

## Checks

- `npm run build`: PASS (Next.js production build and TypeScript check).
- `node --test tests/task8-ui.test.mjs`: PASS (2 cases: streamed briefing/500-character cap/four real agents; backend-down error/no invented briefing; production CSP and security headers).
- Chromium bundle scan for credential patterns and `DISASTERMIND_API_KEY`: PASS (no matches in `.next/static`).
- `python scripts/run_gates.py`: PASS (`59 passed, 1 deselected`; `FINAL_EXIT=0`).
- Red-team review: no client bundle key, simulated timers, fabricated availability values, zero-value substitutes, or unverified agent-completion claims remain in the page. The operator-facing human approval notice is always present.

Task 8 frontend implementation and local checks are complete. Live backend SSE proof is covered separately by Task 7 and remains blocked by upstream NIM availability.
