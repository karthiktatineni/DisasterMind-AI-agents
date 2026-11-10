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
- Bundle scan and browser interaction checks: pending.
- Manual red-team cases: not yet executed; Task 8 remains IN_PROGRESS until these and offline gates are recorded.
