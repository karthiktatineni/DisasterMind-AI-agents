"""Bounded native tool-calling execution loop with round caps, dedup, compaction, and budgets."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from backend.app.orchestration.models import (
    AbstentionItem,
    AgentName,
    AgentStepTrace,
    AgentTraceRecord,
    SharedState,
    SSEEventRecord,
    TokenUsageSummary,
    ToolExecutionEnvelope,
)
from backend.app.orchestration.verifier import (
    AGENT_OUTPUT_MODELS,
    AGENT_REQUIRED_TOOLS,
    InAppVerifier,
)
from backend.app.services.llm_client import LLMClientConfigError, LLMClientError, NativeToolLLMClient
from backend.app.tools.registry import (
    SubmissionToolSpec,
    ToolRegistry,
    compact_tool_result_for_model,
    decode_tool_arguments,
)

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


@dataclass(frozen=True)
class AgentSpec:
    name: AgentName
    prompt_file: str
    allow_list: tuple[str, ...]
    submission_tool_name: str
    submission_description: str
    temperature: float = 0.0
    seed: int = 42
    max_tokens: int = 3200
    max_tool_rounds: int = 3
    max_steps: int = 6
    max_token_budget: int = 24000
    wall_clock_cap_s: float = 150.0
    force_submit_on_required_ok: bool = True

    def load_system_prompt(self) -> str:
        return (PROMPTS_DIR / self.prompt_file).read_text(encoding="utf-8").strip()

    def submission_spec(self) -> SubmissionToolSpec:
        return SubmissionToolSpec(
            name=self.submission_tool_name,
            description=self.submission_description,
            output_model=AGENT_OUTPUT_MODELS[self.name],
        )


def _format_assistant_tool_message(content: str | None, tool_calls: list[Any]) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": content or ""}
    if tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.raw_arguments},
            }
            for tc in tool_calls
        ]
    return msg


def _canonical_dedup_key(tool_name: str, raw_arguments: str) -> tuple[str, str]:
    try:
        parsed = decode_tool_arguments(raw_arguments)
        return tool_name, json.dumps(parsed, sort_keys=True, ensure_ascii=False)
    except Exception:
        return tool_name, raw_arguments.strip()


async def run_agent_loop(
    *,
    spec: AgentSpec,
    user_payload: dict[str, Any],
    state: SharedState,
    llm_client: NativeToolLLMClient,
    registry: ToolRegistry,
    verifier: InAppVerifier,
    emit: Callable[[SSEEventRecord], None],
    run_deadline: float | None = None,
    run_max_tokens: int = 90000,
) -> AgentTraceRecord:
    """Execute one agent via native tools/tool_calls until verified submission, forced submit, or budget cap."""
    started = time.monotonic()
    emit(SSEEventRecord(event="agent_started", run_id=state.run_id, agent=spec.name))

    sub_spec = spec.submission_spec()
    tools_payload = registry.openai_tools_for(spec.allow_list, sub_spec)
    tool_names = [t["function"]["name"] for t in tools_payload]
    system_prompt = spec.load_system_prompt()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]

    steps_trace: list[AgentStepTrace] = []
    called_tools: list[str] = []
    succeeded_tools: set[str] = set()
    rejection_reasons: list[str] = []
    tool_cache: dict[tuple[str, str], ToolExecutionEnvelope] = {}

    total_prompt = 0
    total_comp = 0
    total_reason = 0
    total_toks = 0
    tool_rounds = 0
    repeated_dup_count = 0
    force_submit = False

    def _abstain(reason: str, *, budget_exhausted: bool = False) -> AgentTraceRecord:
        state.abstained_agents.add(spec.name)
        if budget_exhausted:
            state.budget_exhausted_agents.add(spec.name)
        state.abstentions.append(AbstentionItem(agent=spec.name, reason=reason))
        emit(
            SSEEventRecord(
                event="agent_abstained",
                run_id=state.run_id,
                agent=spec.name,
                data={"reason": reason, "budget_exhausted": budget_exhausted},
            )
        )
        return AgentTraceRecord(
            agent=spec.name,
            status="abstained",
            steps=steps_trace,
            called_tools=called_tools,
            rejection_reasons=rejection_reasons,
            total_latency_ms=round((time.monotonic() - started) * 1000.0, 2),
            usage=TokenUsageSummary(
                prompt_tokens=total_prompt,
                completion_tokens=total_comp,
                reasoning_tokens=total_reason,
                total_tokens=total_toks,
            ),
        )

    for step_idx in range(1, spec.max_steps + 1):
        now = time.monotonic()
        if (now - started) >= spec.wall_clock_cap_s:
            return _abstain(f"Agent wall-clock cap ({spec.wall_clock_cap_s}s) reached before verification.", budget_exhausted=True)
        if run_deadline is not None and now >= run_deadline:
            return _abstain("Run-wide wall-clock budget reached before verification.", budget_exhausted=True)
        if total_toks >= spec.max_token_budget:
            return _abstain(f"Agent token budget ({spec.max_token_budget}) reached before verification.", budget_exhausted=True)
        if state.run_total_tokens >= run_max_tokens:
            return _abstain(f"Run-wide token budget ({run_max_tokens}) reached before verification.", budget_exhausted=True)

        choice: str | dict[str, Any] = (
            {"type": "function", "function": {"name": sub_spec.name}}
            if force_submit
            else "required"
        )
        if force_submit:
            messages.append(
                {
                    "role": "user",
                    "content": f"Evidence gathering complete. Call {sub_spec.name} now using ONLY verified tool results.",
                }
            )

        req_snapshot = [dict(m) for m in messages]
        try:
            resp = await llm_client.chat_with_tools(
                messages=messages,
                tools=tools_payload,
                tool_choice=choice,
                temperature=spec.temperature,
                seed=spec.seed,
                max_tokens=spec.max_tokens,
            )
        except (LLMClientConfigError, LLMClientError) as exc:
            rejection_reasons.append(f"LLM unavailable/error: {exc}")
            return _abstain(f"LLM invocation failed: {exc}", budget_exhausted=False)

        if resp.model and resp.model not in state.model_ids:
            state.model_ids.append(resp.model)

        step_usage = TokenUsageSummary(
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            reasoning_tokens=resp.reasoning_tokens,
            total_tokens=resp.total_tokens,
        )
        total_prompt += resp.prompt_tokens
        total_comp += resp.completion_tokens
        total_reason += resp.reasoning_tokens
        total_toks += resp.total_tokens
        state.run_total_tokens += resp.total_tokens

        step_rec = AgentStepTrace(
            step=step_idx,
            request_messages=req_snapshot,
            tools_sent=tool_names,
            model=resp.model,
            temperature=spec.temperature,
            seed=spec.seed,
            finish_reason=resp.finish_reason,
            tool_calls_received=[
                {"id": tc.id, "name": tc.name, "arguments": tc.raw_arguments} for tc in resp.tool_calls
            ],
            assistant_text=resp.content,
            latency_ms=resp.latency_ms,
            usage=step_usage,
        )
        steps_trace.append(step_rec)

        if not resp.tool_calls:
            err = "Control flow requires native tool_calls; received text without tool call."
            rejection_reasons.append(err)
            step_rec.verification_passed = False
            step_rec.verification_errors.append(err)
            emit(
                SSEEventRecord(
                    event="agent_rejected",
                    run_id=state.run_id,
                    agent=spec.name,
                    data={"reason": err, "step": step_idx},
                )
            )
            messages.append({"role": "assistant", "content": resp.content or ""})
            messages.append(
                {
                    "role": "user",
                    "content": f"Rejected: {err} Invoke a tool from {tool_names} via native tool_calls.",
                }
            )
            continue

        messages.append(_format_assistant_tool_message(resp.content, resp.tool_calls))

        new_progress_in_step = False
        has_domain_call = False
        verifier_rejected_this_step = False

        for tc in resp.tool_calls:
            if tc.name == sub_spec.name:
                try:
                    payload = decode_tool_arguments(tc.raw_arguments)
                except (ValueError, json.JSONDecodeError) as exc:
                    err = f"Malformed arguments on submission '{tc.name}': {exc}"
                    rejection_reasons.append(err)
                    step_rec.verification_passed = False
                    step_rec.verification_errors.append(err)
                    verifier_rejected_this_step = True
                    emit(
                        SSEEventRecord(
                            event="agent_rejected",
                            run_id=state.run_id,
                            agent=spec.name,
                            data={"reason": err, "tool": tc.name},
                        )
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps({"status": "rejected", "errors": [err]}),
                        }
                    )
                    continue

                ver = verifier.verify_agent_output(
                    agent=spec.name,
                    raw_payload=payload,
                    agent_called_tools=called_tools,
                    state=state,
                )
                if ver.valid and ver.validated_model is not None:
                    step_rec.verification_passed = True
                    emit(
                        SSEEventRecord(
                            event="agent_verified",
                            run_id=state.run_id,
                            agent=spec.name,
                            data={"tool": tc.name},
                        )
                    )
                    write_rec = state.write_verified(spec.name, spec.name, ver.validated_model)
                    emit(
                        SSEEventRecord(
                            event="state_written",
                            run_id=state.run_id,
                            agent=spec.name,
                            data={"slot": write_rec.agent, "writer": write_rec.writer},
                        )
                    )
                    return AgentTraceRecord(
                        agent=spec.name,
                        status="verified",
                        steps=steps_trace,
                        called_tools=called_tools,
                        rejection_reasons=rejection_reasons,
                        total_latency_ms=round((time.monotonic() - started) * 1000.0, 2),
                        usage=TokenUsageSummary(
                            prompt_tokens=total_prompt,
                            completion_tokens=total_comp,
                            reasoning_tokens=total_reason,
                            total_tokens=total_toks,
                        ),
                    )

                step_rec.verification_passed = False
                step_rec.verification_errors.extend(ver.errors)
                rejection_reasons.extend(ver.errors)
                verifier_rejected_this_step = True
                emit(
                    SSEEventRecord(
                        event="agent_rejected",
                        run_id=state.run_id,
                        agent=spec.name,
                        data={"errors": ver.errors, "tool": tc.name},
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"status": "rejected", "errors": ver.errors}),
                    }
                )
                # If submission was rejected due to a skipped required tool, lift force_submit so agent can execute it.
                if any("Required tool" in e for e in ver.errors):
                    force_submit = False
                continue

            has_domain_call = True
            dedup_key = _canonical_dedup_key(tc.name, tc.raw_arguments)
            if dedup_key in tool_cache:
                cached_env = tool_cache[dedup_key]
                repeated_dup_count += 1
                step_rec.tool_results.append({**cached_env.model_dump(), "deduped": True})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": compact_tool_result_for_model(cached_env, already_provided=True),
                    }
                )
                continue

            emit(
                SSEEventRecord(
                    event="tool_called",
                    run_id=state.run_id,
                    agent=spec.name,
                    data={"call_id": tc.id, "tool": tc.name},
                )
            )
            env = await registry.execute(
                call_id=tc.id,
                agent=spec.name,
                tool_name=tc.name,
                raw_arguments=tc.raw_arguments,
                allow_list=spec.allow_list,
            )
            if env.status == "rejected":
                err = env.error or f"Tool call '{tc.name}' rejected."
                rejection_reasons.append(err)
                step_rec.tool_results.append(env.model_dump())
                emit(
                    SSEEventRecord(
                        event="agent_rejected",
                        run_id=state.run_id,
                        agent=spec.name,
                        data={"call_id": tc.id, "tool": tc.name, "reason": err},
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"status": "rejected", "error": err}),
                    }
                )
                continue

            new_progress_in_step = True
            tool_cache[dedup_key] = env
            called_tools.append(tc.name)
            if env.status == "ok":
                succeeded_tools.add(tc.name)
            state.record_tool_execution(env)
            step_rec.tool_results.append(env.model_dump())
            emit(
                SSEEventRecord(
                    event="tool_result",
                    run_id=state.run_id,
                    agent=spec.name,
                    data={"call_id": tc.id, "tool": tc.name, "status": env.status, "latency_ms": env.latency_ms},
                )
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": compact_tool_result_for_model(env, already_provided=False),
                }
            )

        if has_domain_call and not verifier_rejected_this_step:
            if new_progress_in_step:
                tool_rounds += 1
            req_set = AGENT_REQUIRED_TOOLS.get(spec.name, ())
            all_req_ok = bool(req_set) and all(r in succeeded_tools for r in req_set)
            dup_triggered = (not new_progress_in_step) or (repeated_dup_count > 1)
            all_specialists_attempted = (
                spec.name == "coordinator"
                and all(
                    (s in state.verified_reports or s in state.abstained_agents)
                    for s in ("risk", "logistics", "evacuation")
                )
            )
            if (
                tool_rounds >= spec.max_tool_rounds
                or dup_triggered
                or (spec.force_submit_on_required_ok and all_req_ok)
                or all_specialists_attempted
            ):
                force_submit = True

    return _abstain(f"Exceeded max_steps ({spec.max_steps}) without verified submission.")
