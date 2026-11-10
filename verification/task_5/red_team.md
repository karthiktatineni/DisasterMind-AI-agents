# Task 5 Red-Team & Evidence Self-Check (`ae4503274a41d4c62618470919f35e03c99e1b5b`)

## R3 Test Fixture Edit Record (`tests/test_task4_orchestration.py`, zero assertions weakened)
Reason: Task 5 strict verifier rejects invented narrative numbers (`4-agent` -> `multi-agent`), ungrounded user assumptions (`notes='Pop 100k supplied'` added to 3 fixtures that already passed `100000`), and mismatched mock temperature (`-44.3` -> `31.2` matching `_make_fake_http`). All assertions remain identical.

```diff
diff --git a/tests/test_task4_orchestration.py b/tests/test_task4_orchestration.py
index 061e4c4..7402ee2 100644
--- a/tests/test_task4_orchestration.py
+++ b/tests/test_task4_orchestration.py
@@ -180,7 +180,7 @@ def _happy_handler(*, risk_level: str = "high", evac_rec: str = "mandatory_evacu
                     "coord_4",
                     "submit_coordinator_report",
                     {
-                        "executive_summary": "Verified 4-agent cyclone briefing grounded in Cyclone Fani (2019-0123-IND).",
+                        "executive_summary": "Verified multi-agent cyclone briefing grounded in Cyclone Fani (2019-0123-IND).",
                         "action_plan": ["Initiate coastal shelter activation", "Confirm commander sign-off"],
                         "cited_record_ids": ["2019-0123-IND"],
                         "data_gaps": ["Shelter capacity not provided by user"],
@@ -391,7 +391,7 @@ async def test_task4_scripted_rejection_cases_and_precondition_and_conflict():
             historical_stats_service=_make_fake_stats(),
             http_client=http_client,
         )
-        res = await orch.run({"disaster_type": "Storm", "region": "Odisha", "severity": "Moderate"}, run_id="run-reject-conflict")
+        res = await orch.run({"disaster_type": "Storm", "region": "Odisha", "severity": "Moderate", "notes": "Pop 100k supplied"}, run_id="run-reject-conflict")
 
     assert res.briefing.status == "complete"
     # Conflict between low risk and mandatory_evacuation MUST be flagged explicitly!
@@ -539,7 +539,7 @@ async def test_task4_looping_risk_forced_to_submit_and_deduped():
             http_client=http_client,
             agent_specs=custom_specs,
         )
-        res = await orch.run({"disaster_type": "Storm", "region": "Odisha"}, run_id="run-loop-dedup")
+        res = await orch.run({"disaster_type": "Storm", "region": "Odisha", "notes": "Pop 100k supplied"}, run_id="run-loop-dedup")
 
     assert res.briefing.status == "complete"
     # Historical stats executed only once for Risk despite 2 identical calls (+1 once for Logistics = 2 total).
@@ -592,7 +592,7 @@ async def test_task4_budget_exhaustion_abstained_and_no_redelegation():
             http_client=http_client,
             agent_specs=tiny_budget_specs,
         )
-        res = await orch.run({"disaster_type": "Storm", "region": "Odisha"}, run_id="run-budget-block")
+        res = await orch.run({"disaster_type": "Storm", "region": "Odisha", "notes": "Pop 100k supplied"}, run_id="run-budget-block")
 
     assert risk_invocations["count"] == 1
     assert any(a.agent == "risk" and "token budget" in a.reason.lower() for a in res.briefing.abstentions)
@@ -709,7 +709,7 @@ async def test_task4_no_precedent_with_weather_quantities_and_impossible_premise
                             "premise_valid": True,
                             "hazard_summary": "No precedent; current weather observed.",
                             "cited_record_ids": [],
-                            "quantities": [{"name": "temp", "value": -44.3, "unit": "C", "provenance": "rw1", "inputs": []}],
+                            "quantities": [{"name": "temp", "value": 31.2, "unit": "C", "provenance": "rw1", "inputs": []}],
                             "data_gaps": ["No precedents"],
                         },
                     )
```

## Five Ways Evidence Could Be Fake or Stale - Tested Against `ae4503274a41d4c62618470919f35e03c99e1b5b`
1. **Code changed since verification:** Checked `git status --porcelain -- backend tests config scripts` right before saving evidence -> `""` (`PASS=True`).
2. **Scripted path masking real verifier:** `ScriptedToolLLMClient` supplies only LLM tool-call payloads in offline tests; `InAppVerifier` (`verifier.py` sha256 `7cfe34c3a04ee676...`), `DisasterMindOrchestrator`, and Pydantic `Briefing` run unmodified production code (`PASS=True`).
3. **Wrong Pinecone namespace:** Invoked `verify_runtime_alignment(..., pinecone_namespace='wrong-ns', app_env='production')` and confirmed `StartupIntegrityError` raised (`PASS=True`).
4. **Stale/cached tool result bleeding across runs:** Executed sequential run `rt4-1` (`status=ok`, `1` citation) then `rt4-2` (`status=no_data`); confirmed `rt4-2` returned `status='insufficient_data'`, `0` citations, and Risk abstained without inheriting `rt4-1` cache (`PASS=True`).
5. **Trusting app-written logs without independent re-check:** Verified raw `subprocess` exits from `scripts/run_gates.py` (`exit=0`), raw `git grep` files (`grep_before.txt` / `grep_after.txt`), and fresh `pytest tests/test_task5_grounding.py -v` (`7 passed`, `PASS=True`).
