"""Dependency-free acceptance checker for Task 9 DEV briefings."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any


def check_briefing(value: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return ["briefing must be an object"]
    required = {"run_id", "status", "risk", "evacuation_recommendation", "human_review", "label", "citations", "data_gaps", "disclaimers", "action_plan"}
    missing = sorted(required - value.keys())
    if missing:
        errors.append(f"missing fields: {', '.join(missing)}")
    if not isinstance(value.get("run_id"), str) or not value["run_id"].strip():
        errors.append("run_id must be non-empty")
    if value.get("status") not in {"complete", "partial", "insufficient_data"}:
        errors.append("invalid status")
    if value.get("label") not in {"historical_precedent", "hypothetical_scenario"}:
        errors.append("invalid label")
    risk = value.get("risk")
    if not isinstance(risk, dict) or not isinstance(risk.get("premise_valid"), bool):
        errors.append("risk premise_valid must be boolean")
    evacuation = value.get("evacuation_recommendation")
    if not isinstance(evacuation, dict):
        errors.append("evacuation recommendation must be object")
    elif evacuation.get("requires_human_approval") is not True:
        errors.append("human approval must be required")
    human = value.get("human_review")
    if not isinstance(human, dict) or human.get("required") is not True:
        errors.append("human_review.required must be true")
    citations = value.get("citations")
    if not isinstance(citations, list):
        errors.append("citations must be an array")
    else:
        ids = [item.get("record_id") for item in citations if isinstance(item, dict)]
        if len(ids) != len(citations) or any(not isinstance(item, str) or not item.strip() for item in ids):
            errors.append("each citation must have a record_id")
        if len(set(ids)) != len(ids):
            errors.append("citation record_ids must be unique")
    for field in ("data_gaps", "disclaimers", "action_plan"):
        if not isinstance(value.get(field), list) or any(not isinstance(item, str) for item in value[field]):
            errors.append(f"{field} must be an array of strings")
    if risk and isinstance(risk, dict) and risk.get("premise_valid") is False and value.get("status") == "complete":
        errors.append("invalid premise cannot be marked complete")
    if value.get("status") == "insufficient_data" and citations:
        errors.append("insufficient_data briefing must not claim retrieved citations")
    return errors


def self_test() -> None:
    baseline = {
        "run_id": "dev-self-test", "status": "complete", "label": "hypothetical_scenario",
        "risk": {"premise_valid": True},
        "evacuation_recommendation": {"requires_human_approval": True},
        "human_review": {"required": True}, "citations": [{"record_id": "EMDAT-1"}],
        "data_gaps": [], "disclaimers": [], "action_plan": [],
    }
    assert not check_briefing(baseline), check_briefing(baseline)
    mutations = [
        ("missing run id", lambda x: x.pop("run_id")),
        ("empty run id", lambda x: x.__setitem__("run_id", "")),
        ("unknown status", lambda x: x.__setitem__("status", "success")),
        ("unknown label", lambda x: x.__setitem__("label", "prediction")),
        ("missing risk", lambda x: x.pop("risk")),
        ("nonboolean premise", lambda x: x["risk"].__setitem__("premise_valid", "yes")),
        ("human approval false", lambda x: x["evacuation_recommendation"].__setitem__("requires_human_approval", False)),
        ("review false", lambda x: x["human_review"].__setitem__("required", False)),
        ("citation without id", lambda x: x.__setitem__("citations", [{}])),
        ("duplicate citation", lambda x: x["citations"].append({"record_id": "EMDAT-1"})),
        ("invalid gaps", lambda x: x.__setitem__("data_gaps", "none")),
        ("invalid premise completed", lambda x: x["risk"].__setitem__("premise_valid", False)),
        ("insufficient with citation", lambda x: x.__setitem__("status", "insufficient_data")),
    ]
    for name, mutate in mutations:
        candidate = copy.deepcopy(baseline)
        mutate(candidate)
        if not check_briefing(candidate):
            raise AssertionError(f"mutation escaped checker: {name}")
    print(f"independent checker self-test: PASS ({len(mutations)} rejected mutations)")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        self_test()
    elif len(sys.argv) == 2:
        payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        errors = check_briefing(payload)
        print(json.dumps({"valid": not errors, "errors": errors}, indent=2))
        sys.exit(bool(errors))
    else:
        print("usage: python eval/independent_checker.py --self-test | BRIEFING.json", file=sys.stderr)
        sys.exit(2)
