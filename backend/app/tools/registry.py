"""Tool specification, allow-list filtering, $ref inlining, compacting, validation, and timeout envelopes."""

from __future__ import annotations

import asyncio
import copy
import json
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, ConfigDict, ValidationError

from backend.app.orchestration.models import AgentName, ToolExecutionEnvelope

ToolHandler = Callable[[dict[str, Any], str], Awaitable[tuple[str, dict[str, Any], str | None]]]

MAX_TOOL_RESULT_CHARS = 1800


def inline_json_schema_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve JSON Schema $defs/$ref inline so OpenAI NIM tool parsers never hit unresolved $ref failures."""
    defs = schema.get("$defs") or {}

    def _resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref_key = str(node["$ref"]).split("/")[-1]
                if ref_key in defs:
                    return _resolve(copy.deepcopy(defs[ref_key]))
            return {k: _resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [_resolve(item) for item in node]
        return node

    return _resolve(schema)


def compact_tool_result_for_model(
    envelope: ToolExecutionEnvelope,
    *,
    already_provided: bool = False,
    max_chars: int = MAX_TOOL_RESULT_CHARS,
) -> str:
    """Return bounded summary JSON for the LLM context while preserving full data in traces/SharedState."""
    data = envelope.data
    compact: dict[str, Any] = {
        "status": envelope.status,
        "call_id": envelope.call_id,
    }
    if already_provided:
        compact["already_provided"] = True
    if envelope.error:
        compact["error"] = envelope.error[:240]

    if envelope.tool_name == "search_precedents" and envelope.status == "ok":
        compact["top_score"] = data.get("top_score")
        compact["threshold_used"] = data.get("threshold_used")
        compact["record_ids"] = data.get("record_ids", [])
        compact["matches"] = [
            {
                "record_id": m.get("record_id"),
                "event_name": m.get("event_name"),
                "country": m.get("country"),
                "start_year": m.get("start_year"),
                "total_deaths": m.get("total_deaths"),
                "total_affected": m.get("total_affected"),
                "similarity": m.get("similarity"),
            }
            for m in (data.get("matches") or [])[:5]
        ]
    elif envelope.tool_name == "get_historical_stats" and envelope.status == "ok":
        metrics = data.get("metrics") or {}
        compact["sample_size"] = data.get("sample_size")
        compact["dataset_version"] = data.get("dataset_version")
        compact["metrics"] = {
            k: {"median": v.get("median"), "p90": v.get("p90"), "n": v.get("sample_size")}
            for k, v in metrics.items()
            if isinstance(v, dict)
        }
    else:
        for k, v in data.items():
            if k != "call_id":
                compact[k] = v

    serialized = json.dumps(compact, ensure_ascii=False)
    if len(serialized) <= max_chars:
        return serialized

    # Progressive compaction if still over budget
    if "matches" in compact and isinstance(compact["matches"], list):
        compact["matches"] = compact["matches"][:3]
        compact["truncated"] = True
        serialized = json.dumps(compact, ensure_ascii=False)
    if len(serialized) <= max_chars:
        return serialized

    truncated_summary = {
        "status": envelope.status,
        "call_id": envelope.call_id,
        "already_provided": already_provided,
        "truncated": True,
        "summary": serialized[: max(120, max_chars - 160)],
    }
    return json.dumps(truncated_summary, ensure_ascii=False)[:max_chars]


def decode_tool_arguments(raw_arguments: str | dict[str, Any]) -> dict[str, Any]:
    """Parse OpenAI-protocol tool_calls[].function.arguments JSON payload (never free-text prose)."""
    if isinstance(raw_arguments, dict):
        return raw_arguments
    # Protocol-level JSON decoding strictly for native tool_call arguments.
    parsed = json.loads(raw_arguments)
    if not isinstance(parsed, dict):
        raise ValueError("Tool call arguments must decode to a JSON object.")
    return parsed


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: ToolHandler
    timeout_s: float = 15.0

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": inline_json_schema_refs(self.args_model.model_json_schema()),
            },
        }


class SubmissionToolSpec:
    """Typed final-report submission tool whose inlined JSON schema enforces the agent's Pydantic output."""

    def __init__(self, name: str, description: str, output_model: type[BaseModel]) -> None:
        self.name = name
        self.description = description
        self.output_model = output_model

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": inline_json_schema_refs(self.output_model.model_json_schema()),
            },
        }


class ToolRegistry:
    """Centralized allow-listed executor assigning call_ids, validating schemas, and trapping errors."""

    def __init__(self, specs: list[ToolSpec]) -> None:
        self._specs: dict[str, ToolSpec] = {spec.name: spec for spec in specs}

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def openai_tools_for(self, allow_list: tuple[str, ...] | set[str], submission_tool: SubmissionToolSpec) -> list[dict[str, Any]]:
        tools = [self._specs[name].to_openai_tool() for name in allow_list if name in self._specs]
        tools.append(submission_tool.to_openai_tool())
        return tools

    async def execute(
        self,
        *,
        call_id: str,
        agent: AgentName,
        tool_name: str,
        raw_arguments: str | dict[str, Any],
        allow_list: set[str] | tuple[str, ...],
    ) -> ToolExecutionEnvelope:
        started = time.perf_counter()
        if tool_name not in self._specs:
            return ToolExecutionEnvelope(
                call_id=call_id,
                tool_name=tool_name,
                agent=agent,
                status="rejected",
                error=f"Unknown tool '{tool_name}'.",
                latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
            )
        if tool_name not in allow_list:
            return ToolExecutionEnvelope(
                call_id=call_id,
                tool_name=tool_name,
                agent=agent,
                status="rejected",
                error=f"Tool '{tool_name}' is outside agent '{agent}' allow-list {sorted(allow_list)}.",
                latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
            )

        spec = self._specs[tool_name]
        try:
            decoded = decode_tool_arguments(raw_arguments)
            validated = spec.args_model.model_validate(decoded)
            validated_dict = validated.model_dump()
        except (ValueError, json.JSONDecodeError, ValidationError) as exc:
            return ToolExecutionEnvelope(
                call_id=call_id,
                tool_name=tool_name,
                agent=agent,
                status="rejected",
                error=f"Malformed arguments for '{tool_name}': {exc}",
                latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
            )

        try:
            status, data, err = await asyncio.wait_for(
                spec.handler(validated_dict, call_id),
                timeout=spec.timeout_s,
            )
            return ToolExecutionEnvelope(
                call_id=call_id,
                tool_name=tool_name,
                agent=agent,
                status=status,
                arguments=validated_dict,
                data=data,
                error=err,
                latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
            )
        except asyncio.TimeoutError:
            return ToolExecutionEnvelope(
                call_id=call_id,
                tool_name=tool_name,
                agent=agent,
                status="error",
                arguments=validated_dict,
                error=f"Tool '{tool_name}' timed out after {spec.timeout_s}s.",
                latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
            )
        except Exception as exc:
            return ToolExecutionEnvelope(
                call_id=call_id,
                tool_name=tool_name,
                agent=agent,
                status="error",
                arguments=validated_dict,
                error=f"Tool '{tool_name}' execution failed: {exc}",
                latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
            )


class StrictArgsBase(BaseModel):
    model_config = ConfigDict(extra="forbid")
