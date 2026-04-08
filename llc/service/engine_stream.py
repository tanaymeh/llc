from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_core.messages import HumanMessage

from llc.models import get_model_pricing
from llc.observability import (
    merge_langchain_config,
    set_trace_io,
    start_api_turn_trace,
    update_observation,
)
from llc.service.events import (
    ErrorOccurred,
    SubagentStatusUpdate,
    TextDelta,
    ToolCallStarted,
    ToolResultEvent,
    TurnCompleted,
)
from llc.service.stream_adapter import StreamAdapter

_MAX_TRACE_TEXT_PREVIEW_CHARS = 500


def preview_text(text: str, limit: int = _MAX_TRACE_TEXT_PREVIEW_CHARS) -> str:
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


async def stream_agent_response(
    engine: Any,
    *,
    prompt_text: str,
    turn_id: str,
    persist_user: bool = False,
):
    if engine._agent is None:
        error = ErrorOccurred(message="Agent is unavailable.")
        await engine._persist_event(error, turn_id=turn_id)
        yield error
        return

    if persist_user:
        await engine._persist_message(
            role="user",
            content=prompt_text,
            turn_id=turn_id,
            visible_to_orchestrator=True,
        )

    adapter = StreamAdapter()
    assistant_parts: list[str] = []
    saw_text = False
    last_subagent_snapshot = ""
    tool_output_candidates: dict[str, tuple[str, str]] = {}
    with start_api_turn_trace(
        enabled=engine._langfuse_enabled,
        conversation_id=engine.conversation_id,
        turn_id=turn_id,
        thread_id=engine._thread_id,
        model_name=engine._settings.model_name,
        sub_agent_mode_enabled=engine._settings.sub_agent_mode_enabled,
        prompt_text=prompt_text,
    ) as trace_scope:
        stream_config = merge_langchain_config(
            {"configurable": {"thread_id": engine._thread_id}},
            trace_scope.langchain_config,
        )
        try:
            async for mode, chunk in engine._agent.astream(
                {"messages": [HumanMessage(content=prompt_text)]},
                config=stream_config,
                stream_mode=["messages", "updates"],
            ):
                events = adapter.parse(mode, chunk)
                for event in events:
                    if isinstance(event, TextDelta):
                        saw_text = True
                        assistant_parts.append(event.text)
                    elif isinstance(event, ToolCallStarted):
                        await engine._persist_message(
                            role="assistant",
                            content="",
                            tool_calls=[
                                {
                                    "id": event.tool_call_id,
                                    "type": "tool_call",
                                    "name": event.tool_name,
                                    "args": event.args,
                                }
                            ],
                            turn_id=turn_id,
                            message_kind="tool_call",
                            tool_call_id=event.tool_call_id,
                            visible_to_orchestrator=True,
                        )
                    elif isinstance(event, ToolResultEvent):
                        tool_call_id = event.tool_call_id.strip()
                        if tool_call_id:
                            tool_output_candidates[tool_call_id] = (
                                event.tool_name.strip(),
                                event.content,
                            )
                        await engine._persist_message(
                            role="tool",
                            content=event.content,
                            turn_id=turn_id,
                            message_kind="tool_result",
                            tool_call_id=event.tool_call_id,
                            visible_to_orchestrator=True,
                        )
                    await engine._persist_event(event, turn_id=turn_id)
                    yield event

                if adapter.consume_usage_updated():
                    usage_update = engine._build_usage_event(
                        turn_input_tokens=adapter.turn_input_tokens,
                        turn_output_tokens=adapter.turn_output_tokens,
                        include_pending_turn=True,
                    )
                    await engine._persist_event(usage_update, turn_id=turn_id)
                    yield usage_update

                subagent_event, snapshot_key = engine._build_subagent_status_event(
                    previous_snapshot=last_subagent_snapshot
                )
                if subagent_event is not None and snapshot_key is not None:
                    last_subagent_snapshot = snapshot_key
                    await engine._persist_event(subagent_event, turn_id=turn_id)
                    yield subagent_event
        except asyncio.CancelledError:
            if not engine._interrupt_requested:
                raise
            update_observation(
                trace_scope.observation,
                output={"status": "interrupted", "turn_id": turn_id},
            )
            return
        except Exception as exc:  # noqa: BLE001
            update_observation(
                trace_scope.observation,
                output={"status": "error", "error": str(exc)},
            )
            error = ErrorOccurred(message=str(exc))
            await engine._persist_event(error, turn_id=turn_id)
            yield error
            return

        fallback = adapter.fallback_text.strip()
        if not saw_text and fallback:
            text_event = TextDelta(text=fallback)
            assistant_parts.append(fallback)
            await engine._persist_event(text_event, turn_id=turn_id)
            yield text_event

        turn_input = adapter.turn_input_tokens
        turn_output = adapter.turn_output_tokens
        if turn_input == 0 and turn_output == 0 and adapter.streamed_text_chars > 0:
            turn_output = max(1, adapter.streamed_text_chars // 4)

        hook_message = await engine._run_blocking_hooks(
            turn_input_tokens=turn_input,
            turn_output_tokens=turn_output,
            tool_output_candidates=tool_output_candidates,
        )
        engine._enqueue_background_hooks(
            turn_id=turn_id,
            turn_input_tokens=turn_input,
            turn_output_tokens=turn_output,
            tool_output_candidates=tool_output_candidates,
        )
        usage_final = engine._build_usage_event(
            turn_input_tokens=turn_input,
            turn_output_tokens=turn_output,
            include_pending_turn=False,
        )
        await engine._persist_event(usage_final, turn_id=turn_id)
        yield usage_final

        final_text = "".join(assistant_parts).strip()
        if final_text:
            await engine._persist_message(
                role="assistant",
                content=final_text,
                turn_id=turn_id,
                visible_to_orchestrator=True,
            )

        await engine._persist_usage(
            input_tokens=turn_input,
            output_tokens=turn_output,
        )
        completed = TurnCompleted(
            input_tokens=turn_input,
            output_tokens=turn_output,
            compact_message=hook_message,
        )
        await engine._persist_event(completed, turn_id=turn_id)
        yield completed
        await engine._persist_orchestrator_snapshot()
        update_observation(
            trace_scope.observation,
            output={
                "status": "ok",
                "output_preview": preview_text(final_text),
                "turn_input_tokens": turn_input,
                "turn_output_tokens": turn_output,
            },
        )
        set_trace_io(
            trace_scope.observation,
            input={"user_message": prompt_text},
            output={"assistant_message": final_text},
        )


def build_subagent_status_event(
    engine: Any,
    *,
    previous_snapshot: str,
) -> tuple[SubagentStatusUpdate | None, str | None]:
    runtime = engine._subagent_runtime
    if runtime is None:
        return None, None
    try:
        report = runtime.get_subagent_report(include_all=True)
    except Exception:
        return None, None

    workers = report.get("workers", [])
    if not isinstance(workers, list):
        workers = []
    snapshot_key = json.dumps(
        [
            (
                str(worker.get("id", "")),
                str(worker.get("status", "")),
                float(worker.get("updated_at", 0.0) or 0.0),
            )
            for worker in workers
            if isinstance(worker, dict)
        ],
        sort_keys=True,
        default=str,
    )
    if snapshot_key == previous_snapshot:
        return None, None

    try:
        active_count = int(report.get("active_count", 0) or 0)
    except Exception:
        active_count = 0
    try:
        max_sub_agents = int(report.get("max_sub_agents", 0) or 0)
    except Exception:
        max_sub_agents = 0
    event = SubagentStatusUpdate(
        workers=[worker for worker in workers if isinstance(worker, dict)],
        active_count=max(active_count, 0),
        max_sub_agents=max(max_sub_agents, 0),
    )
    return event, snapshot_key


def build_usage_event(
    engine: Any,
    *,
    turn_input_tokens: int,
    turn_output_tokens: int,
    include_pending_turn: bool,
):
    subagent_input, subagent_output, subagent_cost = engine._subagent_usage_snapshot()
    if include_pending_turn:
        session_input = engine._token_hook.session_input + turn_input_tokens + subagent_input
        session_output = (
            engine._token_hook.session_output + turn_output_tokens + subagent_output
        )
        session_cost = (
            engine._token_hook.session_cost
            + engine._turn_cost(turn_input_tokens, turn_output_tokens)
            + subagent_cost
        )
    else:
        session_input = engine._token_hook.session_input + subagent_input
        session_output = engine._token_hook.session_output + subagent_output
        session_cost = engine._token_hook.session_cost + subagent_cost
    from llc.service.events import UsageUpdate

    return UsageUpdate(
        turn_input_tokens=max(turn_input_tokens, 0),
        turn_output_tokens=max(turn_output_tokens, 0),
        session_input_tokens=max(session_input, 0),
        session_output_tokens=max(session_output, 0),
        session_cost=max(session_cost, 0.0),
    )


def turn_cost(engine: Any, input_tokens: int, output_tokens: int) -> float:
    pricing = get_model_pricing(engine._available_models, engine._settings.model_name)
    if pricing is None:
        return 0.0
    return input_tokens * pricing[0] + output_tokens * pricing[1]


def subagent_usage_snapshot(engine: Any) -> tuple[int, int, float]:
    runtime = engine._subagent_runtime
    if runtime is None:
        return 0, 0, 0.0
    try:
        usage = runtime.get_usage_totals()
    except Exception:
        return 0, 0, 0.0

    input_tokens = max(int(usage.get("input_tokens", 0) or 0), 0)
    output_tokens = max(int(usage.get("output_tokens", 0) or 0), 0)
    by_model = usage.get("by_model", {})
    if not isinstance(by_model, dict):
        return input_tokens, output_tokens, 0.0

    cost = 0.0
    for raw_model_name, raw_bucket in by_model.items():
        model_name = str(raw_model_name).strip()
        if not model_name or not isinstance(raw_bucket, dict):
            continue
        model_input = max(int(raw_bucket.get("input_tokens", 0) or 0), 0)
        model_output = max(int(raw_bucket.get("output_tokens", 0) or 0), 0)
        if model_input == 0 and model_output == 0:
            continue
        pricing = get_model_pricing(engine._available_models, model_name)
        if pricing is None:
            continue
        cost += model_input * pricing[0] + model_output * pricing[1]
    return input_tokens, output_tokens, cost


def terminate_active_subagents(engine: Any, *, reason: str) -> None:
    runtime = engine._subagent_runtime
    if runtime is None:
        return
    try:
        report = runtime.get_subagent_report()
    except Exception:
        return
    workers = report.get("workers", [])
    if not isinstance(workers, list):
        return
    for worker in workers:
        if not isinstance(worker, dict):
            continue
        status = str(worker.get("status", ""))
        if status not in {"running", "restarting", "terminating"}:
            continue
        subagent_id = str(worker.get("id", "")).strip()
        if not subagent_id:
            continue
        try:
            runtime.terminate_subagent(subagent_id, reason=reason)
        except Exception:
            continue


def safe_prompt(engine: Any, name: str, *, key: str, fallback: str) -> str:
    try:
        return engine._prompt_registry.get(name, key=key)
    except Exception:
        return fallback
