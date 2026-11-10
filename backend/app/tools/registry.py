"""Tool specification, allow-list filtering, argument validation, timeout, and error envelopes."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, ConfigDict, ValidationError

from backend.app.orchestration.models import AgentName, ToolExecutionEnvelope

ToolHandler = Callable[[dict[str, Any], str], Awaitable[tuple[str, dict[str, Any], str | None]]]


def _strip_additional_props_default(schema: dict[str, Any]) -> dict[str, Any]:
    """Produce a clean OpenAI-compatible parameter schema from a strict Pydantic model."""
    return schema


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
        schema = _strip_additional_props_default(self.args_model.model_json_schema())
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }


class SubmissionToolSpec:
    """Typed final-report submission tool whose JSON schema enforces the agent's Pydantic output."""

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
                "parameters": self.output_model.model_json_schema(),
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
