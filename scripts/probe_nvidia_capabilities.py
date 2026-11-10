"""Section 2 Pre-Task-4 Capability Probe against NVIDIA NIM (nvidia/nemotron-3-ultra-550b-a55b)."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import get_settings

PROBE_DIR = PROJECT_ROOT / "verification" / "task_4" / "probe"
DEPRECATED_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1"
HARD_REQUEST_BUDGET = 70


class SearchPrecedentsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=3)
    disaster_type: Literal["Flood", "Storm", "Earthquake", "Drought", "Wildfire", "Extreme temperature", "Mass movement (wet)"] | None = None
    country: str | None = None


class GetHistoricalStatsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    disaster_type: Literal["Flood", "Storm", "Earthquake", "Drought", "Wildfire", "Extreme temperature", "Mass movement (wet)"]
    country: str = Field(min_length=2)
    year_min: int = Field(ge=1900, le=2030)
    year_max: int = Field(ge=1900, le=2030)


class GetWeatherArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)


class ProbeBriefSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hazard: str
    severity: Literal["low", "moderate", "high", "extreme"]
    requires_human_approval: bool


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_precedents",
            "description": "Search historical disaster precedents in Pinecone.",
            "parameters": SearchPrecedentsArgs.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_historical_stats",
            "description": "Compute historical aggregate disaster metrics for a country and hazard type.",
            "parameters": GetHistoricalStatsArgs.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Fetch live meteorological forecast from Open-Meteo for latitude/longitude.",
            "parameters": GetWeatherArgs.model_json_schema(),
        },
    },
]

TOOL_ARGS_VALIDATORS: dict[str, type[BaseModel]] = {
    "search_precedents": SearchPrecedentsArgs,
    "get_historical_stats": GetHistoricalStatsArgs,
    "get_weather": GetWeatherArgs,
}


def _safe_headers(headers: httpx.Headers) -> dict[str, str]:
    redacted = {}
    for k, v in headers.items():
        kl = k.lower()
        if kl in ("authorization", "cookie", "set-cookie", "x-api-key"):
            redacted[k] = "[REDACTED]"
        else:
            redacted[k] = v
    return redacted


async def _raw_chat(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    counter: dict[str, int],
) -> dict[str, Any]:
    if counter["count"] >= HARD_REQUEST_BUDGET:
        raise RuntimeError(f"Hard request budget {HARD_REQUEST_BUDGET} reached; aborting live probe.")
    counter["count"] += 1
    url = f"{base_url.rstrip('/')}/chat/completions"
    t0 = time.perf_counter()
    resp = await client.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    try:
        body = resp.json()
    except Exception:
        body = {"raw_text": resp.text}
    return {
        "status_code": resp.status_code,
        "latency_ms": elapsed_ms,
        "headers": _safe_headers(resp.headers),
        "ratelimit_headers": {
            k: v for k, v in resp.headers.items() if "limit" in k.lower() or "retry" in k.lower() or "quota" in k.lower()
        },
        "body": body,
    }


def _extract_choice(res: dict[str, Any]) -> dict[str, Any]:
    body = res.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        return {"error": body, "status_code": res.get("status_code")}
    c0 = choices[0]
    msg = c0.get("message") or {}
    return {
        "status_code": res.get("status_code"),
        "latency_ms": res.get("latency_ms"),
        "model": body.get("model"),
        "system_fingerprint": body.get("system_fingerprint"),
        "finish_reason": c0.get("finish_reason"),
        "content": msg.get("content"),
        "reasoning_content": msg.get("reasoning_content"),
        "tool_calls": msg.get("tool_calls"),
        "usage": body.get("usage"),
        "ratelimit_headers": res.get("ratelimit_headers"),
    }


def _validate_tool_calls(tool_calls: list[dict[str, Any]] | None) -> tuple[bool, list[str], list[dict[str, Any]]]:
    if not tool_calls:
        return False, ["No tool_calls returned"], []
    errors: list[str] = []
    parsed: list[dict[str, Any]] = []
    for tc in tool_calls:
        fn = tc.get("function") or {}
        name = fn.get("name")
        raw_args = fn.get("arguments")
        if name not in TOOL_ARGS_VALIDATORS:
            errors.append(f"Unknown function name: {name!r}")
            continue
        if isinstance(raw_args, dict):
            arg_obj = raw_args
        else:
            try:
                arg_obj = json.loads(raw_args or "")
            except Exception as exc:
                errors.append(f"Invalid JSON arguments for {name}: {exc}")
                continue
        try:
            validated = TOOL_ARGS_VALIDATORS[name].model_validate(arg_obj)
            parsed.append({"id": tc.get("id"), "name": name, "args": validated.model_dump()})
        except ValidationError as exc:
            errors.append(f"Schema validation error for {name}: {exc.errors()}")
    return (len(errors) == 0 and len(parsed) > 0), errors, parsed


async def run_probe() -> dict[str, Any]:
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    base_url = settings.nvidia_base_url.rstrip("/")
    model = settings.nvidia_model
    api_key = settings.nvidia_api_key
    counter = {"count": 0}

    async with httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=20.0)) as client:
        # 1. GET /models and GET /models/{model}
        m_resp = await client.get(f"{base_url}/models", headers={"Authorization": f"Bearer {api_key}"})
        m_detail_resp = await client.get(f"{base_url}/models/{model}", headers={"Authorization": f"Bearer {api_key}"})
        catalog_body = m_resp.json() if m_resp.headers.get("content-type", "").startswith("application/json") else {"raw": m_resp.text}
        detail_body = m_detail_resp.json() if m_detail_resp.headers.get("content-type", "").startswith("application/json") else {"raw": m_detail_resp.text}
        models_list = catalog_body.get("data", []) if isinstance(catalog_body, dict) else []
        target_entry = [m for m in models_list if m.get("id") == model]
        candidates = [
            m.get("id")
            for m in models_list
            if isinstance(m, dict)
            and any(k in str(m.get("id", "")).lower() for k in ("nemotron", "llama-3.3", "qwen", "deepseek", "mistral", "gpt"))
        ]
        (PROBE_DIR / "models_catalog.json").write_text(
            json.dumps(
                {
                    "list_status": m_resp.status_code,
                    "list_headers": _safe_headers(m_resp.headers),
                    "total_models": len(models_list),
                    "target_model_entry": target_entry[0] if target_entry else None,
                    "model_detail_status": m_detail_resp.status_code,
                    "model_detail_body": detail_body,
                    "candidate_alternatives_sample": candidates[:40],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # 2. Native tools & tool_choice (auto, required, named)
        tc_auto = await _raw_chat(
            client,
            base_url,
            api_key,
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a disaster analyst. Use tools to fetch grounded data before answering."},
                    {"role": "user", "content": "Get historical Flood statistics in India from 2010 to 2024."},
                ],
                "tools": TOOL_SCHEMAS,
                "tool_choice": "auto",
                "temperature": 0.0,
                "max_tokens": 1024,
            },
            counter,
        )
        tc_req = await _raw_chat(
            client,
            base_url,
            api_key,
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a disaster analyst."},
                    {"role": "user", "content": "Find historical Storm precedents in Bangladesh."},
                ],
                "tools": TOOL_SCHEMAS,
                "tool_choice": "required",
                "temperature": 0.0,
                "max_tokens": 1024,
            },
            counter,
        )
        tc_named = await _raw_chat(
            client,
            base_url,
            api_key,
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a disaster analyst."},
                    {"role": "user", "content": "Analyze Flood impact in Philippines from 2005 to 2024."},
                ],
                "tools": TOOL_SCHEMAS,
                "tool_choice": {"type": "function", "function": {"name": "get_historical_stats"}},
                "temperature": 0.0,
                "max_tokens": 1024,
            },
            counter,
        )

        # 3. Parallel tool calls
        par_true = await _raw_chat(
            client,
            base_url,
            api_key,
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": "Call every relevant tool immediately in a single parallel batch."},
                    {"role": "user", "content": "Fetch Storm statistics for India 2010-2024 AND fetch live weather at latitude 19.076, longitude 72.877. Invoke both tools now."},
                ],
                "tools": TOOL_SCHEMAS,
                "tool_choice": "auto",
                "parallel_tool_calls": True,
                "temperature": 0.0,
                "max_tokens": 1024,
            },
            counter,
        )
        par_false = await _raw_chat(
            client,
            base_url,
            api_key,
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": "Call tools needed to answer."},
                    {"role": "user", "content": "Fetch Storm statistics for India 2010-2024 AND fetch live weather at latitude 19.076, longitude 72.877."},
                ],
                "tools": TOOL_SCHEMAS,
                "tool_choice": "auto",
                "parallel_tool_calls": False,
                "temperature": 0.0,
                "max_tokens": 1024,
            },
            counter,
        )

        # 4. Seed determinism (2 pairs: JSON prompt + tool-calling prompt)
        seed_payload_json = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Return compact JSON only."},
                {"role": "user", "content": "Classify hazard Cyclone in Odisha, India as JSON matching {hazard, severity, requires_human_approval}."},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 424242,
            "max_tokens": 512,
        }
        seed_json_1 = await _raw_chat(client, base_url, api_key, seed_payload_json, counter)
        seed_json_2 = await _raw_chat(client, base_url, api_key, seed_payload_json, counter)

        seed_payload_tool = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Use tools cleanly."},
                {"role": "user", "content": "Query historical Earthquake stats in Nepal from 2010 to 2024."},
            ],
            "tools": TOOL_SCHEMAS,
            "tool_choice": "required",
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 424242,
            "max_tokens": 1024,
        }
        seed_tool_1 = await _raw_chat(client, base_url, api_key, seed_payload_tool, counter)
        seed_tool_2 = await _raw_chat(client, base_url, api_key, seed_payload_tool, counter)

        # 5. response_format json_object & json_schema
        rf_json_schema = await _raw_chat(
            client,
            base_url,
            api_key,
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": "Produce strictly schema-compliant output."},
                    {"role": "user", "content": "Classify a Category 4 cyclone approaching Visakhapatnam requiring mandatory evacuation."},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ProbeBriefSchema",
                        "strict": True,
                        "schema": ProbeBriefSchema.model_json_schema(),
                    },
                },
                "temperature": 0.0,
                "max_tokens": 1024,
            },
            counter,
        )

        # 6. Context window boundary probe via oversized max_tokens (costs <=5 tokens)
        ctx_probe = await _raw_chat(
            client,
            base_url,
            api_key,
            {
                "model": model,
                "messages": [{"role": "user", "content": "Reply with one word: ok"}],
                "max_tokens": 1_000_000,
                "temperature": 0.0,
            },
            counter,
        )

        # 7. Deprecated model HTTP 410 probe
        res_410 = await _raw_chat(
            client,
            base_url,
            api_key,
            {
                "model": DEPRECATED_MODEL,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 5,
            },
            counter,
        )

        # 8. 20 trials of a simple required tool call
        trial_cases = [
            ("Storm", "India", 2010, 2024),
            ("Flood", "Bangladesh", 2000, 2024),
            ("Earthquake", "Nepal", 2005, 2024),
            ("Wildfire", "United States of America", 2015, 2024),
            ("Drought", "Kenya", 2000, 2024),
            ("Storm", "Philippines", 2008, 2024),
            ("Flood", "Pakistan", 2010, 2024),
            ("Earthquake", "Japan", 2000, 2024),
            ("Extreme temperature", "France", 2000, 2024),
            ("Mass movement (wet)", "Colombia", 2000, 2024),
            ("Storm", "Viet Nam", 2010, 2024),
            ("Flood", "Nigeria", 2012, 2024),
            ("Earthquake", "Türkiye", 2000, 2024),
            ("Wildfire", "Australia", 2010, 2024),
            ("Drought", "Ethiopia", 2000, 2024),
            ("Storm", "Mexico", 2010, 2024),
            ("Flood", "Brazil", 2010, 2024),
            ("Earthquake", "Chile", 2000, 2024),
            ("Storm", "Myanmar", 2005, 2024),
            ("Flood", "Indonesia", 2005, 2024),
        ]
        trials_results: list[dict[str, Any]] = []
        valid_count = 0
        for idx, (hazard, country, y0, y1) in enumerate(trial_cases, 1):
            raw = await _raw_chat(
                client,
                base_url,
                api_key,
                {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "Call get_historical_stats with exact fields."},
                        {"role": "user", "content": f"Retrieve historical {hazard} stats for {country} between {y0} and {y1}."},
                    ],
                    "tools": TOOL_SCHEMAS,
                    "tool_choice": "required",
                    "temperature": 0.0,
                    "max_tokens": 1024,
                },
                counter,
            )
            ch = _extract_choice(raw)
            ok, errs, parsed = _validate_tool_calls(ch.get("tool_calls"))
            if ok:
                valid_count += 1
            trials_results.append(
                {
                    "trial": idx,
                    "prompt_inputs": {"hazard": hazard, "country": country, "year_min": y0, "year_max": y1},
                    "status_code": raw["status_code"],
                    "latency_ms": raw["latency_ms"],
                    "finish_reason": ch.get("finish_reason"),
                    "valid_tool_call": ok,
                    "errors": errs,
                    "parsed_tool_calls": parsed,
                    "raw_tool_calls": ch.get("tool_calls"),
                    "reasoning_chars": len(ch.get("reasoning_content") or ""),
                    "usage": ch.get("usage"),
                    "ratelimit_headers": raw.get("ratelimit_headers"),
                }
            )

        # 9. Controlled RPM burst (20 rapid concurrent max_tokens=1 pings to inspect 429 + headers)
        burst_payload = {
            "model": model,
            "messages": [{"role": "user", "content": "ok"}],
            "max_tokens": 1,
            "temperature": 0.0,
        }
        t_burst0 = time.perf_counter()
        burst_resps = await asyncio.gather(
            *[_raw_chat(client, base_url, api_key, burst_payload, counter) for _ in range(20)],
            return_exceptions=True,
        )
        burst_wall_s = round(time.perf_counter() - t_burst0, 3)
        burst_summary = []
        for idx, br in enumerate(burst_resps, 1):
            if isinstance(br, Exception):
                burst_summary.append({"i": idx, "exception": repr(br)})
            else:
                burst_summary.append(
                    {
                        "i": idx,
                        "status_code": br["status_code"],
                        "latency_ms": br["latency_ms"],
                        "ratelimit_headers": br["ratelimit_headers"],
                        "error_body": br["body"] if br["status_code"] != 200 else None,
                    }
                )

    c_auto = _extract_choice(tc_auto)
    c_req = _extract_choice(tc_req)
    c_named = _extract_choice(tc_named)
    c_par_true = _extract_choice(par_true)
    c_par_false = _extract_choice(par_false)
    c_sj1 = _extract_choice(seed_json_1)
    c_sj2 = _extract_choice(seed_json_2)
    c_st1 = _extract_choice(seed_tool_1)
    c_st2 = _extract_choice(seed_tool_2)
    c_rf_schema = _extract_choice(rf_json_schema)

    schema_valid = False
    schema_err = None
    try:
        if c_rf_schema.get("content"):
            ProbeBriefSchema.model_validate_json(c_rf_schema["content"])
            schema_valid = True
    except Exception as exc:
        schema_err = str(exc)

    json_obj_valid = False
    try:
        if c_sj1.get("content"):
            json.loads(c_sj1["content"])
            json_obj_valid = True
    except Exception:
        pass

    valid_rate = round(valid_count / len(trial_cases), 4)
    status_429_count = sum(1 for b in burst_summary if b.get("status_code") == 429)

    report = {
        "model": model,
        "base_url": base_url,
        "total_live_chat_requests": counter["count"],
        "native_tools": {
            "auto_valid": _validate_tool_calls(c_auto.get("tool_calls"))[0],
            "auto_choice": c_auto,
            "required_valid": _validate_tool_calls(c_req.get("tool_calls"))[0],
            "required_choice": c_req,
            "named_valid": _validate_tool_calls(c_named.get("tool_calls"))[0],
            "named_choice": c_named,
        },
        "parallel_tool_calls": {
            "parallel_true_count": len(c_par_true.get("tool_calls") or []),
            "parallel_true_valid": _validate_tool_calls(c_par_true.get("tool_calls"))[0],
            "parallel_true_calls": c_par_true.get("tool_calls"),
            "parallel_false_count": len(c_par_false.get("tool_calls") or []),
            "parallel_false_calls": c_par_false.get("tool_calls"),
        },
        "seed_determinism": {
            "json_content_identical": c_sj1.get("content") == c_sj2.get("content"),
            "json_reasoning_identical": c_sj1.get("reasoning_content") == c_sj2.get("reasoning_content"),
            "json_run1": c_sj1,
            "json_run2": c_sj2,
            "tool_calls_identical": c_st1.get("tool_calls") == c_st2.get("tool_calls"),
            "tool_args_identical": (
                [(p["name"], p["args"]) for p in _validate_tool_calls(c_st1.get("tool_calls"))[2]]
                == [(p["name"], p["args"]) for p in _validate_tool_calls(c_st2.get("tool_calls"))[2]]
            ),
            "tool_run1": c_st1,
            "tool_run2": c_st2,
        },
        "response_format": {
            "json_object_status": seed_json_1.get("status_code"),
            "json_object_valid_json": json_obj_valid,
            "json_schema_status": rf_json_schema.get("status_code"),
            "json_schema_strict_valid": schema_valid,
            "json_schema_error": schema_err,
            "json_schema_choice": c_rf_schema,
        },
        "context_window_probe": {
            "status_code": ctx_probe.get("status_code"),
            "body": ctx_probe.get("body"),
            "headers": ctx_probe.get("headers"),
        },
        "deprecated_410_probe": {
            "model": DEPRECATED_MODEL,
            "status_code": res_410.get("status_code"),
            "body": res_410.get("body"),
            "ratelimit_headers": res_410.get("ratelimit_headers"),
        },
        "required_tool_20_trials": {
            "total": len(trial_cases),
            "valid": valid_count,
            "rate": valid_rate,
            "gate_95_percent_pass": valid_rate >= 0.95,
        },
        "rpm_and_429_burst": {
            "burst_concurrency": 20,
            "burst_wall_seconds": burst_wall_s,
            "status_200_count": sum(1 for b in burst_summary if b.get("status_code") == 200),
            "status_429_count": status_429_count,
            "burst_results": burst_summary,
        },
    }

    (PROBE_DIR / "probe_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (PROBE_DIR / "probe_trials_20.json").write_text(json.dumps(trials_results, indent=2), encoding="utf-8")
    print(
        f"[PROBE DONE] total_chat_calls={counter['count']} valid_tool_rate={valid_count}/20 ({valid_rate:.1%}) "
        f"parallel={report['parallel_tool_calls']['parallel_true_count']} "
        f"seed_json_eq={report['seed_determinism']['json_content_identical']} "
        f"seed_tool_eq={report['seed_determinism']['tool_args_identical']} "
        f"json_schema={schema_valid} status_410={res_410['status_code']} burst_429={status_429_count}"
    )
    return report


async def run_phase2() -> dict[str, Any]:
    """Run targeted seed@T=0.7 check (3 calls) + 45-ping RPM/429 boundary test (48 calls total)."""
    import argparse
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    base_url = settings.nvidia_base_url.rstrip("/")
    model = settings.nvidia_model
    api_key = settings.nvidia_api_key
    counter = {"count": 0}

    async with httpx.AsyncClient(timeout=httpx.Timeout(45.0, connect=15.0)) as client:
        # 1. Seed behavior at temperature=0.7 (2 identical seed=424242, 1 different seed=999999)
        base_seed_req = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Return compact JSON {city, hazard, score} only."},
                {"role": "user", "content": "Pick one random Asian coastal city, one hazard, and an integer score 1-100."},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.7,
            "top_p": 1.0,
            "max_tokens": 256,
        }
        s1 = await _raw_chat(client, base_url, api_key, {**base_seed_req, "seed": 424242}, counter)
        s2 = await _raw_chat(client, base_url, api_key, {**base_seed_req, "seed": 424242}, counter)
        s3 = await _raw_chat(client, base_url, api_key, {**base_seed_req, "seed": 999999}, counter)

        # 2. 45-request rapid concurrent wave (max_tokens=1) to probe 40 RPM limit & exact 429 payload/headers
        ping = {"model": model, "messages": [{"role": "user", "content": "ok"}], "max_tokens": 1, "temperature": 0.0}
        t0 = time.perf_counter()
        resps = await asyncio.gather(*[_raw_chat(client, base_url, api_key, ping, counter) for _ in range(45)], return_exceptions=True)
        wall_s = round(time.perf_counter() - t0, 3)

    c1, c2, c3 = _extract_choice(s1), _extract_choice(s2), _extract_choice(s3)
    status_counts: dict[str, int] = {}
    sample_429 = None
    ratelimit_headers_seen = []
    for br in resps:
        if isinstance(br, Exception):
            k = type(br).__name__
            status_counts[k] = status_counts.get(k, 0) + 1
        else:
            sc = str(br["status_code"])
            status_counts[sc] = status_counts.get(sc, 0) + 1
            if br.get("ratelimit_headers"):
                ratelimit_headers_seen.append({"status": br["status_code"], "headers": br["ratelimit_headers"]})
            if br["status_code"] == 429 and sample_429 is None:
                sample_429 = {"status_code": 429, "headers": br["headers"], "body": br["body"]}

    # Also patch tool_args_identical in existing probe_report.json using exact saved data
    rep_path = PROBE_DIR / "probe_report.json"
    if rep_path.exists():
        existing = json.loads(rep_path.read_text(encoding="utf-8"))
        st1 = existing["seed_determinism"]["tool_run1"].get("tool_calls")
        st2 = existing["seed_determinism"]["tool_run2"].get("tool_calls")
        existing["seed_determinism"]["tool_args_identical"] = (
            [(p["name"], p["args"]) for p in _validate_tool_calls(st1)[2]]
            == [(p["name"], p["args"]) for p in _validate_tool_calls(st2)[2]]
        )
        rep_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    out = {
        "phase2_calls": counter["count"],
        "seed_at_temp_0_7": {
            "seed424242_run1_content": c1.get("content"),
            "seed424242_run2_content": c2.get("content"),
            "seed999999_run3_content": c3.get("content"),
            "same_seed_content_identical": c1.get("content") == c2.get("content"),
            "same_seed_reasoning_identical": c1.get("reasoning_content") == c2.get("reasoning_content"),
        },
        "rpm_45_wave": {
            "concurrency": 45,
            "wall_seconds": wall_s,
            "status_counts": status_counts,
            "ratelimit_headers_seen": ratelimit_headers_seen[:5],
            "sample_429": sample_429,
        },
    }
    (PROBE_DIR / "probe_rpm_seed_phase2.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("[PHASE2 DONE]", json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--phase2", action="store_true")
    args = p.parse_args()
    asyncio.run(run_phase2() if args.phase2 else run_probe())

