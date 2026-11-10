"""Bounded, raw-evidence live checks for Task 7 HTTP and SSE endpoints."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "verification" / "final_checks"
BASE = "http://127.0.0.1:8000"


def record_response(path: str, request_id: str, body: dict) -> dict:
    started = time.perf_counter()
    with httpx.Client(timeout=httpx.Timeout(480.0, connect=10.0)) as client:
        response = client.post(
            BASE + path,
            json=body,
            headers={"X-Request-ID": request_id, "Accept": "text/event-stream" if path.endswith("/stream") else "application/json"},
        )
    elapsed = round((time.perf_counter() - started) * 1000, 3)
    raw = response.text
    (OUT / f"{request_id}.response.txt").write_text(raw, encoding="utf-8")
    result = {
        "path": path,
        "request_id": request_id,
        "status_code": response.status_code,
        "response_request_id": response.headers.get("x-request-id"),
        "content_type": response.headers.get("content-type"),
        "elapsed_ms": elapsed,
        "response_bytes": len(response.content),
    }
    if path.endswith("/stream"):
        events = []
        for block in raw.strip().split("\n\n"):
            entry = {}
            for line in block.splitlines():
                if line.startswith("event: "):
                    entry["event"] = line[7:]
                elif line.startswith("data: "):
                    try:
                        entry["data"] = json.loads(line[6:])
                    except json.JSONDecodeError:
                        entry["data"] = line[6:]
            if entry:
                events.append(entry)
        result["events"] = events
        result["event_types"] = [e.get("event") for e in events]
        result["event_runs_match_request_id"] = all(
            e.get("data", {}).get("run_id") == request_id for e in events if isinstance(e.get("data"), dict)
        )
    else:
        try:
            decoded = response.json()
        except ValueError:
            decoded = None
        result["body"] = decoded
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", choices=("orchestrate", "stream", "both"), default="both")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=10.0) as client:
        health = client.get(BASE + "/health", headers={"X-Request-ID": "task7-health-real"})
        health_result = {"status_code": health.status_code, "request_id": health.headers.get("x-request-id"), "body": health.json()}
    report = {"health": health_result}
    if args.endpoint in ("orchestrate", "both"):
        report["orchestrate"] = record_response(
            "/orchestrate",
            "task7-orchestrate-real-001",
            {"disaster_type": "Flood", "region": "Sylhet, Bangladesh", "severity": "High", "time_horizon": "48 hours"},
        )
    if args.endpoint in ("stream", "both"):
        report["stream"] = record_response(
            "/orchestrate/stream",
            "task7-stream-real-001",
            {"disaster_type": "Storm", "region": "Odisha, India", "severity": "High", "time_horizon": "48 hours", "notes": "Population is unspecified; do not infer it."},
        )
    target = OUT / "task7_live_http_report.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "health_status": health_result["status_code"],
        "orchestrate_status": report.get("orchestrate", {}).get("status_code"),
        "orchestrate_request_id_matches": report.get("orchestrate", {}).get("response_request_id") == report.get("orchestrate", {}).get("request_id"),
        "stream_status": report.get("stream", {}).get("status_code"),
        "stream_events": report.get("stream", {}).get("event_types", []),
        "stream_run_ids_match": report.get("stream", {}).get("event_runs_match_request_id"),
        "evidence": str(target.relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
