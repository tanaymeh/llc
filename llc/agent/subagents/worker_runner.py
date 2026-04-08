from __future__ import annotations

import asyncio
import json
import time
from contextlib import suppress
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from llc.agent.message_utils import message_text
from llc.agent.subagents.coordination import AgentCoordinationLayer
from llc.config import Settings
from llc.logging_utils import configure_logging
from llc.observability import (
    flush,
    merge_langchain_config,
    start_linked_subagent_trace,
    update_observation,
)
from llc.service.prompt_registry import PromptRegistry
from llc.agent.subagents.reporting import activity_from_tool_name, preview_text, stop_reason_is_stuck
from llc.agent.subagents.usage import (
    accumulate_usage_by_model,
    normalize_usage_by_model,
    token_usage,
)

_MAX_STORED_FINAL_OUTPUT_CHARS = 12000
_MAX_REPORT_PREVIEW_CHARS = 220
_MAX_FINAL_OUTPUT_PREVIEW_CHARS = 600
_MAX_REVISION_PART_CHARS = 1800
_MAX_TOOL_SUMMARY_CHARS = 140
_STREAM_POLL_INTERVAL_S = 1.0
_STREAM_CLEANUP_TIMEOUT_S = 5.0
_INTERNAL_TOOL_NAMES = {
    "SendMessage",
    "ReadInbox",
    "ReadTeamStatus",
    "ReadSharedNotes",
    "AppendSharedNote",
    "RequestLock",
    "ReleaseLock",
    "ReviewHeldLocks",
    "RespondLockReview",
    "LaunchSubagent",
    "GetSubagentReport",
    "WaitSubagents",
    "ReviseSubagent",
    "InterruptSubagent",
    "TerminateSubagent",
}
_COORDINATION_EVENT_TOOL_NAMES = {
    "SendMessage",
    "ReadInbox",
    "ReadTeamStatus",
    "ReadSharedNotes",
    "AppendSharedNote",
    "RequestLock",
    "ReleaseLock",
    "ReviewHeldLocks",
    "RespondLockReview",
}


class StopRequested(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _compact_path(raw_path: str) -> str:
    cleaned = raw_path.replace("\\", "/").strip()
    if len(cleaned) <= 64:
        return cleaned
    parts = [part for part in cleaned.split("/") if part]
    if len(parts) >= 2:
        tail = "/".join(parts[-2:])
        if len(tail) <= 48:
            return f".../{tail}"
    return cleaned[:61] + "..."


def _tool_summary(tool_name: str, args: dict[str, Any]) -> str:
    name = tool_name.strip()
    if not name or not isinstance(args, dict):
        return ""

    if name in {"Read", "Write", "Edit", "MultiEdit"}:
        path = str(args.get("file_path") or args.get("path") or "").strip()
        if path:
            return f"path={_compact_path(path)}"
    if name == "LS":
        path = str(args.get("path") or "").strip()
        return f"path={_compact_path(path)}" if path else "workspace"
    if name == "Bash":
        cmd = str(args.get("command") or "").strip()
        if cmd:
            return preview_text(cmd, _MAX_TOOL_SUMMARY_CHARS)
    if name == "TodoWrite":
        todos = args.get("todos")
        if isinstance(todos, list):
            return f"todo_items={len(todos)}"
    if name == "SendMessage":
        recipient = str(args.get("recipient_subagent_id") or "").strip()
        content = str(args.get("content") or "").strip()
        recipient_part = f"to={recipient} " if recipient else ""
        if content:
            return f"{recipient_part}msg={preview_text(content, 70)}".strip()
        return recipient_part.strip()
    if name in {"RequestLock", "ReleaseLock"}:
        path = str(args.get("file_path") or "").strip()
        if path:
            return f"path={_compact_path(path)}"
    if name in {"ReadInbox", "ReadTeamStatus", "ReadSharedNotes", "ReviewHeldLocks"}:
        return "coordination_check"
    return ""


def subagent_process_entry(
    *,
    payload: dict[str, Any],
    stop_event: Any,
    worker_queue: Any,
) -> None:
    attempt = int(payload.get("attempt", 0) or 0)
    try:
        settings_payload = payload.get("settings")
        settings = Settings.model_validate(
            settings_payload if isinstance(settings_payload, dict) else {}
        )
        configure_logging(subagent_debug_logging=settings.sub_agent_debug_logging)
        subagent_id = str(payload.get("subagent_id", "")).strip()
        task = str(payload.get("task", "")).strip()
        context = str(payload.get("context", "")).strip()
        thread_id = str(payload.get("thread_id", "")).strip()
        subagent_name = str(payload.get("subagent_name", "")).strip()
        parent_context_raw = payload.get("parent_context")
        parent_context = (
            parent_context_raw if isinstance(parent_context_raw, dict) else None
        )
        coordination_handles_raw = payload.get("coordination_handles")
        coordination_handles = (
            coordination_handles_raw
            if isinstance(coordination_handles_raw, dict)
            else None
        )
        coordination_layer = (
            AgentCoordinationLayer.from_handles(coordination_handles)
            if coordination_handles is not None
            else None
        )
        prompt_registry = PromptRegistry(settings.prompts_dir)
        from llc.agent.graph import build_agent_graph

        agent = build_agent_graph(
            settings,
            role="subagent",
            subagent_coordination=coordination_layer,
            subagent_id=subagent_id,
        )
        result = asyncio.run(
            run_subagent_worker(
                agent=agent,
                task=task,
                context=context,
                thread_id=thread_id,
                subagent_id=subagent_id,
                subagent_name=subagent_name,
                settings=settings,
                prompt_registry=prompt_registry,
                parent_context=parent_context,
                stop_requested=lambda: stop_event_is_set(stop_event),
                event_callback=lambda event: emit_worker_event(
                    worker_queue,
                    {
                        "attempt": attempt,
                        **event,
                    },
                ),
            )
        )
    except Exception as exc:  # noqa: BLE001
        result = {
            "status": "failed",
            "error": str(exc),
            "tool_calls": 0,
            "total_tool_calls": 0,
            "output_chars": 0,
            "usage_by_model": {},
        }
    emit_worker_event(
        worker_queue,
        {
            "type": "result",
            "attempt": attempt,
            "result": result,
        },
    )


async def run_subagent_worker(
    *,
    agent: Any,
    task: str,
    context: str,
    thread_id: str,
    subagent_id: str,
    subagent_name: str,
    settings: Settings,
    prompt_registry: PromptRegistry | None,
    parent_context: dict[str, Any] | None,
    stop_requested: Callable[[], bool],
    event_callback: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    started_monotonic = time.monotonic()
    report_interval = float(settings.sub_agent_report_interval_s)
    max_runtime_s = float(settings.sub_agent_max_runtime_s)
    llm_stall_timeout_s = float(settings.sub_agent_llm_stall_timeout_s)
    tool_stall_timeout_s = float(settings.sub_agent_tool_stall_timeout_s)
    if llm_stall_timeout_s > 0 and llm_stall_timeout_s >= max_runtime_s:
        llm_stall_timeout_s = max(max_runtime_s * 0.25, 30.0)
    if tool_stall_timeout_s > 0 and tool_stall_timeout_s >= max_runtime_s:
        tool_stall_timeout_s = max(max_runtime_s * 0.5, 30.0)
    max_tool_calls = int(settings.sub_agent_max_tool_calls)
    heartbeat_interval_s = min(max(report_interval / 4.0, 0.5), 2.0)
    next_report_at = started_monotonic + report_interval
    next_heartbeat_at = started_monotonic + heartbeat_interval_s
    last_chunk_at = started_monotonic
    tool_calls = 0
    total_tool_calls = 0
    output_chars = 0
    final_parts: list[str] = []
    fallback_text = ""
    usage_by_model: dict[str, dict[str, int]] = {}
    emitted_usage_by_model: dict[str, dict[str, int]] = {}
    last_tool_name = ""
    last_tool_summary = ""
    saw_any_tool_call = False
    last_completed_node = ""
    current_activity = "Running"
    activity_detail = "Worker started."
    seen_tool_call_ids: set[str] = set()
    seen_tool_result_ids: set[str] = set()
    final_history: list[dict[str, Any]] = []

    def emit(event_type: str, **payload: Any) -> None:
        event_callback(
            {
                "type": event_type,
                **payload,
            }
        )

    def emit_status(*, include_report: bool) -> None:
        payload = {
            "tool_calls": tool_calls,
            "total_tool_calls": total_tool_calls,
            "output_chars": output_chars,
            "activity": current_activity,
            "activity_detail": activity_detail,
            "last_tool_name": last_tool_name,
            "last_tool_summary": last_tool_summary,
        }
        if include_report:
            payload["report"] = (
                "Running: "
                f"tool_calls={tool_calls}, "
                f"total_tool_calls={total_tool_calls}, "
                f"output_chars={output_chars}."
            )
        emit("status", **payload)
        flush()

    def emit_usage_delta() -> None:
        usage_delta = _usage_delta(
            current_usage=usage_by_model,
            emitted_usage=emitted_usage_by_model,
        )
        if not usage_delta:
            return
        emit("usage", usage_by_model=usage_delta)
        for model_name, bucket in usage_delta.items():
            emitted_bucket = emitted_usage_by_model.setdefault(
                model_name,
                {"input_tokens": 0, "output_tokens": 0},
            )
            emitted_bucket["input_tokens"] += int(bucket.get("input_tokens", 0) or 0)
            emitted_bucket["output_tokens"] += int(bucket.get("output_tokens", 0) or 0)

    with start_linked_subagent_trace(
        parent_context=parent_context,
        subagent_id=subagent_id,
        subagent_name=subagent_name,
        thread_id=thread_id,
        task=task,
        model_name=settings.model_name,
    ) as trace_scope:
        trace_stream_config = dict(trace_scope.langchain_config)
        stream_config = merge_langchain_config(
            {"configurable": {"thread_id": thread_id}},
            trace_stream_config,
        )
        stream = agent.astream(
            {
                "messages": [
                    SystemMessage(
                        content=render_subagent_system_prompt(
                            prompt_registry,
                        )
                    ),
                    HumanMessage(
                        content=render_subagent_task_message(
                            task,
                            context,
                            prompt_registry,
                        )
                    ),
                ]
            },
            config=stream_config,
            stream_mode=["messages", "updates"],
        )
        pending_chunk: asyncio.Task[Any] | None = None
        try:
            try:
                while True:
                    if stop_requested():
                        raise StopRequested("stop_requested")
                    now_monotonic = time.monotonic()
                    elapsed = now_monotonic - started_monotonic
                    if elapsed > max_runtime_s:
                        raise StopRequested("max_runtime_exceeded")
                    in_tool_phase = last_completed_node == "llm"
                    active_stall_timeout = (
                        tool_stall_timeout_s if in_tool_phase
                        else llm_stall_timeout_s
                    )
                    if (
                        active_stall_timeout > 0
                        and (now_monotonic - last_chunk_at) > active_stall_timeout
                    ):
                        raise StopRequested("stall_timeout_exceeded")
                    next_chunk_timeout_s = next_stream_poll_timeout_s(
                        elapsed=elapsed,
                        max_runtime_s=max_runtime_s,
                    )
                    if next_chunk_timeout_s <= 0:
                        raise StopRequested("max_runtime_exceeded")
                    if pending_chunk is None:
                        pending_chunk = asyncio.create_task(anext(stream))
                    try:
                        mode, chunk = await asyncio.wait_for(
                            asyncio.shield(pending_chunk),
                            timeout=next_chunk_timeout_s,
                        )
                        pending_chunk = None
                    except StopAsyncIteration:
                        pending_chunk = None
                        break
                    except asyncio.TimeoutError:
                        continue

                    last_chunk_at = time.monotonic()

                    if mode == "messages":
                        msg_chunk, meta = chunk
                        if meta.get("langgraph_node") == "llm":
                            text = message_text(msg_chunk.content)
                            if text:
                                final_parts.append(text)
                                output_chars += len(text)

                    if mode == "updates" and isinstance(chunk, dict):
                        for node_name, node_update in chunk.items():
                            if isinstance(node_name, str) and node_name.strip():
                                last_completed_node = node_name.strip()
                            if not isinstance(node_update, dict):
                                continue
                            for message in node_update.get("messages", []):
                                if isinstance(message, AIMessage):
                                    tool_calls_batch = (
                                        getattr(message, "tool_calls", None) or []
                                    )
                                    maybe_text = message_text(message.content)
                                    if maybe_text:
                                        fallback_text = maybe_text
                                        if not tool_calls_batch:
                                            current_activity = "Analyzing task"
                                            activity_detail = preview_text(
                                                maybe_text,
                                                _MAX_REPORT_PREVIEW_CHARS,
                                            )

                                    for raw_tool_call in tool_calls_batch:
                                        if not isinstance(raw_tool_call, dict):
                                            continue
                                        tool_call_id = str(
                                            raw_tool_call.get("id", "") or ""
                                        ).strip()
                                        if not tool_call_id or tool_call_id in seen_tool_call_ids:
                                            continue
                                        seen_tool_call_ids.add(tool_call_id)
                                        saw_any_tool_call = True
                                        total_tool_calls += 1
                                        if (
                                            str(raw_tool_call.get("name", "")).strip()
                                            not in _INTERNAL_TOOL_NAMES
                                        ):
                                            tool_calls += 1
                                        if tool_calls > max_tool_calls:
                                            raise StopRequested("max_tool_calls_exceeded")
                                        raw_tool_name = raw_tool_call.get("name", "")
                                        if (
                                            isinstance(raw_tool_name, str)
                                            and raw_tool_name.strip()
                                        ):
                                            last_tool_name = raw_tool_name.strip()
                                            raw_args = raw_tool_call.get("args", {})
                                            args = raw_args if isinstance(raw_args, dict) else {}
                                            last_tool_summary = _tool_summary(
                                                last_tool_name,
                                                args,
                                            )
                                            current_activity = activity_from_tool_name(
                                                last_tool_name
                                            )
                                            if last_tool_summary:
                                                activity_detail = (
                                                    f"Using {last_tool_name}: {last_tool_summary}"
                                                )
                                            else:
                                                activity_detail = (
                                                    f"Using {last_tool_name}."
                                                )
                                            emit(
                                                "tool_started",
                                                tool_name=last_tool_name,
                                                tool_call_id=tool_call_id,
                                                args=args,
                                                activity=current_activity,
                                            )
                                            if (
                                                last_tool_name
                                                in _COORDINATION_EVENT_TOOL_NAMES
                                            ):
                                                emit(
                                                    "coordination_event",
                                                    coordination_event_type=(
                                                        "tool_started"
                                                    ),
                                                    tool_name=last_tool_name,
                                                    tool_call_id=tool_call_id,
                                                    args=args,
                                                )
                                            flush()

                                    usage_input, usage_output = token_usage(message)
                                    accumulate_usage_by_model(
                                        usage_by_model,
                                        settings.model_name,
                                        usage_input,
                                        usage_output,
                                    )
                                    emit_usage_delta()
                                    continue

                                if not isinstance(message, ToolMessage):
                                    continue
                                tool_call_id = str(
                                    getattr(message, "tool_call_id", "") or ""
                                ).strip()
                                if (
                                    not tool_call_id
                                    or tool_call_id in seen_tool_result_ids
                                ):
                                    continue
                                seen_tool_result_ids.add(tool_call_id)
                                additional_kwargs = (
                                    getattr(message, "additional_kwargs", {}) or {}
                                )
                                tool_name = str(
                                    additional_kwargs.get("tool_name", "tool") or "tool"
                                ).strip()
                                tool_content = message_text(message.content)
                                emit(
                                    "tool_completed",
                                    tool_name=tool_name,
                                    tool_call_id=tool_call_id,
                                    content=tool_content,
                                )
                                coordination_event = _coordination_event_from_tool_result(
                                    tool_name=tool_name,
                                    tool_call_id=tool_call_id,
                                    args={},
                                    content=tool_content,
                                )
                                if coordination_event is not None:
                                    emit("coordination_event", **coordination_event)
                                flush()

                    now_monotonic = time.monotonic()
                    if now_monotonic >= next_report_at:
                        emit_status(include_report=True)
                        next_report_at = now_monotonic + report_interval
                        next_heartbeat_at = now_monotonic + heartbeat_interval_s
                    elif now_monotonic >= next_heartbeat_at:
                        emit_status(include_report=False)
                        next_heartbeat_at = now_monotonic + heartbeat_interval_s
            finally:
                if pending_chunk is not None:
                    pending_chunk.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await asyncio.wait_for(
                            asyncio.shield(pending_chunk),
                            timeout=_STREAM_CLEANUP_TIMEOUT_S,
                        )
                aclose = getattr(stream, "aclose", None)
                if callable(aclose):
                    try:
                        await asyncio.wait_for(
                            aclose(),
                            timeout=_STREAM_CLEANUP_TIMEOUT_S,
                        )
                    except (asyncio.TimeoutError, Exception):
                        pass
        except StopRequested as exc:
            status = "stuck" if stop_reason_is_stuck(exc.reason) else "terminated"
            final_history = await _serialize_worker_history(agent, thread_id)
            update_observation(
                trace_scope.observation,
                output={"status": status, "stop_reason": exc.reason},
            )
            flush()
            return {
                "status": status,
                "stop_reason": exc.reason,
                "tool_calls": tool_calls,
                "total_tool_calls": total_tool_calls,
                "output_chars": output_chars,
                "usage_by_model": usage_by_model,
                "history": final_history,
            }
        except Exception as exc:  # noqa: BLE001
            final_history = await _serialize_worker_history(agent, thread_id)
            update_observation(
                trace_scope.observation,
                output={"status": "failed", "error": str(exc)},
            )
            flush()
            return {
                "status": "failed",
                "error": str(exc),
                "tool_calls": tool_calls,
                "total_tool_calls": total_tool_calls,
                "output_chars": output_chars,
                "usage_by_model": usage_by_model,
                "history": final_history,
            }

        final_output = "".join(final_parts).strip()
        if not final_output:
            final_output = fallback_text.strip()
        if len(final_output) > _MAX_STORED_FINAL_OUTPUT_CHARS:
            final_output = final_output[: _MAX_STORED_FINAL_OUTPUT_CHARS - 3] + "..."
        if not saw_any_tool_call:
            if settings.sub_agent_require_tool_call:
                error_message = (
                    "Sub-agent finished without calling any tools. "
                    "Treating this as a no-op worker result."
                )
                update_observation(
                    trace_scope.observation,
                    output={
                        "status": "failed",
                        "error": error_message,
                        "final_output_preview": preview_text(
                            final_output,
                            _MAX_FINAL_OUTPUT_PREVIEW_CHARS,
                        ),
                    },
                )
                flush()
                return {
                    "status": "failed",
                    "error": error_message,
                    "tool_calls": tool_calls,
                    "total_tool_calls": total_tool_calls,
                    "output_chars": output_chars,
                    "usage_by_model": usage_by_model,
                    "history": await _serialize_worker_history(agent, thread_id),
                }
            if not final_output:
                error_message = (
                    "Sub-agent finished without calling tools and produced no final output."
                )
                update_observation(
                    trace_scope.observation,
                    output={
                        "status": "failed",
                        "error": error_message,
                    },
                )
                flush()
                return {
                    "status": "failed",
                    "error": error_message,
                    "tool_calls": tool_calls,
                    "total_tool_calls": total_tool_calls,
                    "output_chars": output_chars,
                    "usage_by_model": usage_by_model,
                    "history": await _serialize_worker_history(agent, thread_id),
                }
        final_history = await _serialize_worker_history(agent, thread_id)
        result = {
            "status": "completed",
            "final_output": final_output,
            "tool_calls": tool_calls,
            "total_tool_calls": total_tool_calls,
            "output_chars": output_chars,
            "usage_by_model": usage_by_model,
            "history": final_history,
        }
        update_observation(
            trace_scope.observation,
            output={
                "status": "completed",
                "final_output_preview": preview_text(
                    final_output,
                    _MAX_FINAL_OUTPUT_PREVIEW_CHARS,
                ),
            },
        )
        flush()
        return result


def emit_worker_event(worker_queue: Any, event: dict[str, Any]) -> None:
    try:
        worker_queue.put_nowait(event)
    except Exception:
        pass


def stop_event_is_set(stop_event: Any) -> bool:
    if stop_event is None:
        return False
    try:
        return bool(stop_event.is_set())
    except Exception:
        return False


def render_subagent_system_prompt(
    prompt_registry: PromptRegistry | None = None,
) -> str:
    if prompt_registry is not None:
        try:
            rendered = prompt_registry.get(
                "subagent_mode",
                key="subagent_mode_prompt",
            ).strip()
            if rendered:
                return rendered
        except Exception:
            pass
    return ""


def render_subagent_task_message(
    task: str,
    context: str,
    prompt_registry: PromptRegistry | None = None,
) -> str:
    clean_task = task.strip()
    clean_context = context.strip()
    context_block = ""
    if clean_context:
        context_block = (
            "\n\n"
            "Additional context and instructions from orchestrator:\n"
            f"{clean_context}"
        )
    if prompt_registry is not None:
        try:
            rendered = prompt_registry.get_formatted(
                "subagent_mode",
                key="task_message_prompt",
                task=clean_task,
                context_block=context_block,
            ).strip()
            if rendered:
                return rendered
        except Exception:
            pass
    return f"Here's your task:\n{clean_task}{context_block}"


def task_with_feedback(
    base_task: str,
    feedback: str,
    prior_output: str,
    prompt_registry: PromptRegistry | None = None,
) -> str:
    base = base_task.strip()
    prior = prior_output.strip()
    revised = feedback.strip()
    trimmed_prior = preview_text(prior, _MAX_REVISION_PART_CHARS) if prior else ""
    trimmed_feedback = preview_text(revised, _MAX_REVISION_PART_CHARS) if revised else ""

    if prompt_registry is not None:
        try:
            rendered = prompt_registry.get_formatted(
                "subagent_mode",
                key="revision_task_prompt",
                base_task=base,
                prior_output=trimmed_prior,
                feedback=trimmed_feedback,
            ).strip()
            if rendered:
                return rendered
        except Exception:
            pass

    if not revised and not prior:
        return base_task
    parts = [f"Original task:\n{base}"]
    if prior:
        parts.append(f"Prior attempt output:\n{trimmed_prior}")
    if revised:
        parts.append(f"Revision instructions:\n{trimmed_feedback}")
    parts.append("Produce an improved final result.")
    return "\n\n".join(parts)


def next_stream_poll_timeout_s(
    *,
    elapsed: float,
    max_runtime_s: float,
) -> float:
    remaining_runtime = max(max_runtime_s - elapsed, 0.0)
    return min(remaining_runtime, _STREAM_POLL_INTERVAL_S)


async def _serialize_worker_history(agent: Any, thread_id: str) -> list[dict[str, Any]]:
    try:
        state = await agent.aget_state({"configurable": {"thread_id": thread_id}})
    except Exception:
        return []
    values = getattr(state, "values", {})
    messages = values.get("messages", [])
    if not isinstance(messages, list):
        return []
    return _serialized_history_messages(messages)


def _serialized_history_messages(messages: list[Any]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    saw_agent_reply = False
    for message in messages:
        if isinstance(message, AIMessage):
            saw_agent_reply = True
        if not saw_agent_reply and isinstance(message, (SystemMessage, HumanMessage)):
            continue
        item = _serialize_history_message(message)
        if item is not None:
            serialized.append(item)
    return serialized


def _serialize_history_message(message: Any) -> dict[str, Any] | None:
    if isinstance(message, HumanMessage):
        return {
            "role": "user",
            "content": message_text(message.content),
            "message_kind": "history_snapshot",
        }
    if isinstance(message, SystemMessage):
        return {
            "role": "system",
            "content": message_text(message.content),
            "message_kind": "history_snapshot",
        }
    if isinstance(message, AIMessage):
        raw_tool_calls = getattr(message, "tool_calls", None) or []
        tool_calls = raw_tool_calls if isinstance(raw_tool_calls, list) else []
        return {
            "role": "assistant",
            "content": message_text(message.content),
            "tool_calls": tool_calls,
            "message_kind": "history_snapshot",
        }
    if isinstance(message, ToolMessage):
        tool_call_id = str(getattr(message, "tool_call_id", "") or "").strip()
        additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
        return {
            "role": "tool",
            "content": message_text(message.content),
            "tool_call_id": tool_call_id,
            "tool_name": str(additional_kwargs.get("tool_name", "") or "").strip(),
            "message_kind": "history_snapshot",
        }
    return None


def _usage_delta(
    *,
    current_usage: dict[str, dict[str, int]],
    emitted_usage: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    delta: dict[str, dict[str, int]] = {}
    normalized_current = normalize_usage_by_model(current_usage)
    normalized_emitted = normalize_usage_by_model(emitted_usage)
    for model_name, bucket in normalized_current.items():
        current_input = int(bucket.get("input_tokens", 0) or 0)
        current_output = int(bucket.get("output_tokens", 0) or 0)
        emitted_bucket = normalized_emitted.get(model_name, {})
        emitted_input = int(emitted_bucket.get("input_tokens", 0) or 0)
        emitted_output = int(emitted_bucket.get("output_tokens", 0) or 0)
        delta_input = max(current_input - emitted_input, 0)
        delta_output = max(current_output - emitted_output, 0)
        if delta_input == 0 and delta_output == 0:
            continue
        delta[model_name] = {
            "input_tokens": delta_input,
            "output_tokens": delta_output,
        }
    return delta


def _coordination_event_from_tool_result(
    *,
    tool_name: str,
    tool_call_id: str,
    args: dict[str, Any],
    content: str,
) -> dict[str, Any] | None:
    clean_tool_name = tool_name.strip()
    if clean_tool_name not in _COORDINATION_EVENT_TOOL_NAMES:
        return None
    payload = _parse_json_payload(content)
    event: dict[str, Any] = {
        "coordination_event_type": clean_tool_name.lower(),
        "tool_name": clean_tool_name,
        "tool_call_id": tool_call_id,
    }
    if args:
        event["args"] = args
    if payload is not None:
        event["payload"] = payload
    return event


def _parse_json_payload(content: str) -> dict[str, Any] | None:
    clean_content = content.strip()
    if not clean_content:
        return None
    try:
        parsed = json.loads(clean_content)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed
