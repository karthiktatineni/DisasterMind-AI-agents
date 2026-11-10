"""One-shot NIM availability probe with secret-safe HTTP evidence."""
from __future__ import annotations

import asyncio
from datetime import datetime
import json
import re
from pathlib import Path
import sys
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "verification" / "final_checks" / "nim_availability.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.config import get_settings
SAFE_HEADERS = {
    "date", "retry-after", "retry-after-ms", "content-type", "x-request-id",
    "request-id", "x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset",
    "server", "via",
}
SECRET_RE = re.compile(r"(?i)(nvapi-[A-Za-z0-9_-]{10,}|pcsk_[A-Za-z0-9_-]{10,}|sb_secret_[A-Za-z0-9_-]{10,})")


def _safe_body(text: str) -> str:
    return SECRET_RE.sub("[REDACTED_SECRET]", text[:4000])


async def main() -> None:
    settings = get_settings()
    endpoint = settings.nvidia_base_url.rstrip("/") + "/chat/completions"
    probes = [
        ("tiny_chat", {
            "model": settings.nvidia_model,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "max_tokens": 128,
            "temperature": 0,
            "stream": False,
        }),
        ("forced_tool", {
            "model": settings.nvidia_model,
            "messages": [
                {"role": "system", "content": "When requested, call the required function with the specified arguments."},
                {"role": "user", "content": "Call probe with ok=true."},
            ],
            "tools": [{"type": "function", "function": {
                "name": "probe", "description": "Availability probe only.",
                "parameters": {"type": "object", "properties": {"ok": {"type": "boolean"}},
                               "required": ["ok"], "additionalProperties": False},
            }}],
            "tool_choice": "required",
            "max_tokens": 512,
            "temperature": 0,
            "stream": False,
        }),
    ]
    previous = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    results: list[dict[str, Any]] = list(previous.get("probes", []))
    starting_count = len(results)
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        for name, payload in probes:
            started = datetime.now().astimezone().isoformat()
            try:
                response = await client.post(
                    endpoint,
                    headers={"Authorization": f"Bearer {settings.nvidia_api_key}", "Content-Type": "application/json"},
                    json=payload,
                )
                response_json = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
                choices = response_json.get("choices") or []
                message = choices[0].get("message", {}) if choices else {}
                tool_calls = message.get("tool_calls") or []
                results.append({
                    "probe": name,
                    "timestamp": started,
                    "status": response.status_code,
                    "headers": {k.lower(): v for k, v in response.headers.items() if k.lower() in SAFE_HEADERS},
                    "body": _safe_body(response.text),
                    "finish_reason": choices[0].get("finish_reason") if choices else None,
                    "tool_call_observed": bool(tool_calls) if name == "forced_tool" else None,
                    "tool_call_names": [item.get("function", {}).get("name") for item in tool_calls],
                    "model": settings.nvidia_model,
                    "base_host": httpx.URL(settings.nvidia_base_url).host,
                })
                if response.status_code == 429:
                    break
            except Exception as exc:  # only class is retained; exception text can include request details
                results.append({"probe": name, "timestamp": started, "error_type": type(exc).__name__,
                                "model": settings.nvidia_model, "base_host": httpx.URL(settings.nvidia_base_url).host})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                               "probes": results}, indent=2), encoding="utf-8")
    print(json.dumps(results[starting_count:], indent=2))


if __name__ == "__main__":
    asyncio.run(main())
