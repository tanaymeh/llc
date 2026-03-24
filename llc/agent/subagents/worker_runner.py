from __future__ import annotations

import asyncio
import logging
import time
import traceback
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from llc.agent.message_utils import message_text
from llc.agent.subagents.coordination import SubAgentCoordinationClient
from llc.agent.subagents.reporting import activity_from_tool_name, preview_text, stop_reason_is_stuck
from llc.agent.subagents.usage import accumulate_usage_by_model, token_usage
from llc.config import Settings
from llc.logging import configure_logging
from llc.observability import (
    merge_langchain_config,
    start_linked_subagent_trace,
    update_observation,
)
from llc.service.prompt_registry import PromptRegistry

_MAX_STORED_FINAL_OUTPUT_CHARS = 12000
_MAX_REPORT_PREVIEW_CHARS = 220
_MAX_FINAL_OUTPUT_PREVIEW_CHARS = 600
_MAX_REVISION_PART_CHARS = 1800

logger = logging.getLogger(__name__)


class StopRequested(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def subagent_process_entry(
    *,
    payload: dict[str, Any],
    stop_event: Any,
    worker_queue: Any,
) -> None:
    configure_logging()
    attempt = int(payload.get("attempt", 0) or 0)
    subagent_id = str(payload.get("subagent_id", "")).strip()
    subagent_name = str(payload.get("subagent_name", "")).strip()
    try:
        settings_payload = payload.get("settings")
        settings = Settings.model_validate(
            settings_payload if isinstance(settings_payload, dict) else {}
        )
        task = str(payload.get("task", "")).strip()
        context = str(payload.get("context", "")).strip()
        thread_id = str(payload.get("thread_id", "")).strip()
        coordination_payload = payload.get("coordination")
        coordination_client = (
            SubAgentCoordinationClient.from_payload(coordination_payload)
            if isinstance(coordination_payload, dict)
            else None
        )
        parent_context_raw = payload.get("parent_context")
        parent_context = (
            parent_context_raw if isinstance(parent_context_raw, dict) else None
        )
        prompt_registry = PromptRegistry(settings.prompts_dir)
        from llc.agent.graph import build_agent_graph

        agent = build_agent_graph(
            settings,
            role="subagent",
            subagent_coordination=coordination_client,
            prompt_registry=prompt_registry,
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
                coordination_client=coordination_client,
                parent_context=parent_context,
                stop_requested=lambda: stop_event_is_set(stop_event),
                progress_callback=lambda progress: emit_worker_event(
                    worker_queue,
                    {
                        "type": "progress",
                        "attempt": attempt,
                        **progress,
                    },
                ),
            )
        )
    except Exception as exc:  # noqa: BLE001
        error_traceback = traceback.format_exc()
        logger.exception(
            "Sub-agent worker crashed before returning a result: id=%s name=%s attempt=%s",
            subagent_id or "unknown",
            subagent_name or "unknown",
            attempt,
        )
        result = {
            "status": "failed",
            "error": str(exc),
            "error_traceback": error_traceback,
            "tool_calls": 0,
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
    coordination_client: SubAgentCoordinationClient | None,
    parent_context: dict[str, Any] | None,
    stop_requested: Callable[[], bool],
    progress_callback: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    started_monotonic = time.monotonic()
    report_interval = float(settings.sub_agent_report_interval_s)
    max_runtime_s = float(settings.sub_agent_max_runtime_s)
    stall_timeout_s = float(settings.sub_agent_stall_timeout_s)
    max_tool_calls = int(settings.sub_agent_max_tool_calls)
    heartbeat_interval_s = min(max(report_interval / 4.0, 0.5), 2.0)
    next_report_at = started_monotonic + report_interval
    next_heartbeat_at = started_monotonic + heartbeat_interval_s
    tool_calls = 0
    output_chars = 0
    final_parts: list[str] = []
    fallback_text = ""
    usage_by_model: dict[str, dict[str, int]] = {}
    last_tool_name = ""
    current_activity = "Running"
    activity_detail = "Worker started."

    # Mutable state shared with the background heartbeat task so it can
    # send up-to-date snapshots even when the stream loop is blocked.
    _shared: dict[str, Any] = {
        "tool_calls": 0,
        "output_chars": 0,
        "activity": current_activity,
        "activity_detail": activity_detail,
        "last_tool_name": "",
    }

    async def _background_heartbeat() -> None:
        while True:
            await asyncio.sleep(heartbeat_interval_s)
            progress_callback({
                "tool_calls": _shared["tool_calls"],
                "output_chars": _shared["output_chars"],
                "activity": _shared["activity"],
                "activity_detail": _shared["activity_detail"],
                "last_tool_name": _shared["last_tool_name"],
            })

    heartbeat_task: asyncio.Task[None] | None = None

    task_message = render_subagent_task_message(
        task,
        context,
        prompt_registry,
    )
    if coordination_client is not None:
        coordination_client.set_waiting_on("")
        coordination_state = coordination_client.state()
        task_message = add_coordination_block(task_message, coordination_state)
    with start_linked_subagent_trace(
        parent_context=parent_context,
        subagent_id=subagent_id,
        subagent_name=subagent_name,
        thread_id=thread_id,
        task=task,
        model_name=settings.model_name,
    ) as trace_scope:
        trace_stream_config = dict(trace_scope.langchain_config)
        trace_stream_config.pop("callbacks", None)
        stream_config = merge_langchain_config(
            {"configurable": {"thread_id": thread_id}},
            trace_stream_config,
        )
        stream = agent.astream(
            {
                "messages": [
                    SystemMessage(
                        content=render_subagent_system_prompt(
                            settings,
                            prompt_registry,
                        )
                    ),
                    HumanMessage(
                        content=task_message
                    ),
                ]
            },
            config=stream_config,
            stream_mode=["messages", "updates"],
        )
        heartbeat_task = asyncio.create_task(_background_heartbeat())
        try:
            try:
                while True:
                    if stop_requested():
                        raise StopRequested("stop_requested")
                    elapsed = time.monotonic() - started_monotonic
                    if elapsed > max_runtime_s:
                        raise StopRequested("max_runtime_exceeded")

                    is_waiting_on_peer = (
                        coordination_client is not None
                        and bool(coordination_client.get_waiting_on())
                    )
                    effective_stall = stall_timeout_s
                    if is_waiting_on_peer:
                        effective_stall = max_runtime_s - elapsed

                    next_chunk_timeout_s = next_stream_timeout_s(
                        elapsed=elapsed,
                        max_runtime_s=max_runtime_s,
                        stall_timeout_s=effective_stall,
                    )
                    if next_chunk_timeout_s <= 0:
                        raise StopRequested("max_runtime_exceeded")
                    try:
                        mode, chunk = await asyncio.wait_for(
                            anext(stream),
                            timeout=next_chunk_timeout_s,
                        )
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        raise StopRequested("stall_timeout_exceeded")

                    if mode == "messages":
                        msg_chunk, meta = chunk
                        if meta.get("langgraph_node") == "llm":
                            text = message_text(msg_chunk.content)
                            if text:
                                final_parts.append(text)
                                output_chars += len(text)

                    if mode == "updates" and isinstance(chunk, dict):
                        for node_update in chunk.values():
                            if not isinstance(node_update, dict):
                                continue
                            for message in node_update.get("messages", []):
                                if not isinstance(message, AIMessage):
                                    continue
                                tool_calls_batch = getattr(message, "tool_calls", None) or []
                                tool_calls += len(tool_calls_batch)
                                if tool_calls > max_tool_calls:
                                    raise StopRequested("max_tool_calls_exceeded")
                                if tool_calls_batch:
                                    raw_tool_name = tool_calls_batch[-1].get("name", "")
                                    if (
                                        isinstance(raw_tool_name, str)
                                        and raw_tool_name.strip()
                                    ):
                                        last_tool_name = raw_tool_name.strip()
                                        current_activity = activity_from_tool_name(
                                            last_tool_name
                                        )
                                        if last_tool_name == "WaitForPeerMessage":
                                            activity_detail = "Waiting for peer input."
                                        else:
                                            activity_detail = f"Using {last_tool_name}."
                                usage_input, usage_output = token_usage(message)
                                accumulate_usage_by_model(
                                    usage_by_model,
                                    settings.model_name,
                                    usage_input,
                                    usage_output,
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

                    _shared["tool_calls"] = tool_calls
                    _shared["output_chars"] = output_chars
                    _shared["activity"] = current_activity
                    _shared["activity_detail"] = activity_detail
                    _shared["last_tool_name"] = last_tool_name

                    now_monotonic = time.monotonic()
                    if now_monotonic >= next_report_at:
                        progress_callback(
                            {
                                "tool_calls": tool_calls,
                                "output_chars": output_chars,
                                "report": (
                                    "Running: "
                                    f"tool_calls={tool_calls}, output_chars={output_chars}."
                                ),
                                "activity": current_activity,
                                "activity_detail": activity_detail,
                                "last_tool_name": last_tool_name,
                            }
                        )
                        next_report_at = now_monotonic + report_interval
                        next_heartbeat_at = now_monotonic + heartbeat_interval_s
                    elif now_monotonic >= next_heartbeat_at:
                        progress_callback(
                            {
                                "tool_calls": tool_calls,
                                "output_chars": output_chars,
                                "activity": current_activity,
                                "activity_detail": activity_detail,
                                "last_tool_name": last_tool_name,
                            }
                        )
                        next_heartbeat_at = now_monotonic + heartbeat_interval_s
            finally:
                if heartbeat_task is not None:
                    heartbeat_task.cancel()
                    try:
                        await heartbeat_task
                    except (asyncio.CancelledError, Exception):
                        pass
                aclose = getattr(stream, "aclose", None)
                if callable(aclose):
                    try:
                        await aclose()
                    except Exception:
                        pass
        except StopRequested as exc:
            status = "stuck" if stop_reason_is_stuck(exc.reason) else "terminated"
            update_observation(
                trace_scope.observation,
                output={"status": status, "stop_reason": exc.reason},
            )
            return {
                "status": status,
                "stop_reason": exc.reason,
                "tool_calls": tool_calls,
                "output_chars": output_chars,
                "usage_by_model": usage_by_model,
            }
        except Exception as exc:  # noqa: BLE001
            error_traceback = traceback.format_exc()
            logger.exception(
                "Sub-agent worker failed during execution: id=%s name=%s",
                subagent_id or "unknown",
                subagent_name or "unknown",
            )
            update_observation(
                trace_scope.observation,
                output={"status": "failed", "error": str(exc)},
            )
            return {
                "status": "failed",
                "error": str(exc),
                "error_traceback": error_traceback,
                "tool_calls": tool_calls,
                "output_chars": output_chars,
                "usage_by_model": usage_by_model,
            }

        final_output = "".join(final_parts).strip()
        if not final_output:
            final_output = fallback_text.strip()
        if len(final_output) > _MAX_STORED_FINAL_OUTPUT_CHARS:
            final_output = final_output[: _MAX_STORED_FINAL_OUTPUT_CHARS - 3] + "..."
        result = {
            "status": "completed",
            "final_output": final_output,
            "tool_calls": tool_calls,
            "output_chars": output_chars,
            "usage_by_model": usage_by_model,
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
    settings: Settings,
    prompt_registry: PromptRegistry | None = None,
) -> str:
    base_prompt = settings.system_prompt.rstrip()
    if prompt_registry is not None:
        try:
            rendered = prompt_registry.get_formatted(
                "subagent_mode",
                key="subagent_mode_prompt",
                base_prompt=base_prompt,
            ).strip()
            if rendered:
                return rendered
        except Exception:
            pass
    return base_prompt


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


def add_coordination_block(task_message: str, coordination_state: dict[str, Any]) -> str:
    if not isinstance(coordination_state, dict) or not coordination_state:
        return task_message
    worker_id = str(coordination_state.get("worker_id", "")).strip()
    unread_count = int(coordination_state.get("unread_count", 0) or 0)
    plan_submitted = bool(coordination_state.get("plan_submitted", False))
    team_directory_raw = coordination_state.get("team_directory")
    lines = [
        "Coordination context:",
        f"- Worker ID: {worker_id or 'unknown'}",
        f"- Inbox unread messages: {max(unread_count, 0)}",
        f"- Plan submitted: {'yes' if plan_submitted else 'no'}",
    ]
    if isinstance(team_directory_raw, list):
        lines.append("- Team directory:")
        for item in team_directory_raw[:8]:
            if not isinstance(item, dict):
                continue
            peer_id = str(item.get("id", "")).strip()
            name = str(item.get("name", "")).strip()
            goal = str(item.get("goal", "")).strip()
            active = bool(item.get("active", False))
            waiting_on = str(item.get("waiting_on", "")).strip()
            label = f"{name} ({peer_id})" if name and peer_id else peer_id or name or "worker"
            status = "active" if active else "inactive"
            waiting_text = f"; waiting_on={waiting_on}" if waiting_on else ""
            goal_text = goal if goal else "n/a"
            lines.append(f"  - {label}: {status}{waiting_text}; goal={goal_text}")
    block = "\n".join(lines).strip()
    if not block:
        return task_message
    return f"{task_message}\n\n{block}"


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


def next_stream_timeout_s(
    *,
    elapsed: float,
    max_runtime_s: float,
    stall_timeout_s: float,
) -> float:
    remaining_runtime = max(max_runtime_s - elapsed, 0.0)
    if stall_timeout_s <= 0:
        return remaining_runtime
    return min(remaining_runtime, stall_timeout_s)
