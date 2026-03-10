from __future__ import annotations

import asyncio
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from threading import RLock
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from llc.agent.llm import build_chat_model
from llc.agent.message_utils import message_text
from llc.agent.subagents.types import ACTIVE_STATUSES, SubAgentRecord
from llc.config import Settings

SubAgentBuilder = Callable[[Settings], Any]
_MAX_STORED_FINAL_OUTPUT_CHARS = 12000
_MAX_TASK_PREVIEW_CHARS = 180
_MAX_REPORT_PREVIEW_CHARS = 220
_MAX_ERROR_PREVIEW_CHARS = 220
_MAX_FINAL_OUTPUT_PREVIEW_CHARS = 600
_MAX_REVISION_PART_CHARS = 1800
_MAX_REPORT_HISTORY_CHARS = 28000
_MAX_HISTORY_MESSAGE_CHARS = 1000
_MAX_REPORT_INPUT_FINAL_CHARS = 2400
_MAX_COMPLETION_REPORT_CHARS = 900
_REPORT_LLM_TIMEOUT_S = 8.0
_MAX_REPORT_WORKERS = 8
_SUBAGENT_REPORT_PROMPT = (
    "You are writing a concise, information-rich execution report for a completed coding task.\n"
    "Return 4 short sections using plain text headings:\n"
    "Goal\n"
    "Work done\n"
    "Outcome\n"
    "Notes\n"
    "Keep total output under a 1000 words.\n"
    "Do not include chain-of-thought or unnecessary detail."
)


class _StopRequested(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class SubAgentRuntime:
    def __init__(self, settings: Settings, build_subagent: SubAgentBuilder) -> None:
        self._settings = settings
        self._build_subagent = build_subagent
        self._records: dict[str, SubAgentRecord] = {}
        self._lock = RLock()
        self._closed = False
        self._executor = ThreadPoolExecutor(
            max_workers=settings.max_sub_agents,
            thread_name_prefix="llc-subagent",
        )

    def update_settings(self, settings: Settings) -> None:
        with self._lock:
            self._settings = settings

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for record in self._records.values():
                if record.status in ACTIVE_STATUSES:
                    record.terminate_requested = True
                    record.stop_reason = "runtime_shutdown"
                    record.status = "terminating"
                    record.stop_event.set()
            self._executor.shutdown(wait=False, cancel_futures=True)

    def launch_subagent(
        self,
        task: str,
        *,
        context: str = "",
        source: str = "orchestrator",
        name: str | None = None,
    ) -> dict[str, Any]:
        clean_task = task.strip()
        if not clean_task:
            return {"ok": False, "error": "Task cannot be empty."}

        with self._lock:
            if self._closed:
                return {"ok": False, "error": "Sub-agent runtime is not available."}
            self._refresh_stuck_locked()
            if self._active_count_locked() >= self._settings.max_sub_agents:
                return {
                    "ok": False,
                    "error": (
                        f"Maximum active sub-agents reached "
                        f"({self._settings.max_sub_agents})."
                    ),
                }

            subagent_id = f"subagent-{uuid.uuid4().hex[:8]}"
            now = time.time()
            record = SubAgentRecord(
                id=subagent_id,
                name=name or subagent_id,
                base_task=clean_task,
                task=clean_task,
                context=context.strip(),
                source=source,
                thread_id=f"{subagent_id}-{uuid.uuid4().hex[:6]}",
                created_at=now,
                updated_at=now,
                latest_report="Queued.",
            )
            self._records[subagent_id] = record
            self._submit_locked(record)
            snapshot = self._record_snapshot_locked(record)
            snapshot["ok"] = True
            return snapshot

    def get_subagent_report(self, ids: list[str] | None = None) -> dict[str, Any]:
        with self._lock:
            self._refresh_stuck_locked()
            selected = self._select_records_for_report_locked(ids)
            workers = [self._record_snapshot_locked(record) for record in selected]
            return {
                "ok": True,
                "active_count": self._active_count_locked(),
                "max_sub_agents": self._settings.max_sub_agents,
                "workers": workers,
            }

    def wait_subagents(
        self,
        ids: list[str] | None = None,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        timeout_s = 0.0
        if timeout_ms and timeout_ms > 0:
            timeout_s = timeout_ms / 1000
        deadline = time.monotonic() + timeout_s if timeout_s > 0 else None

        while True:
            with self._lock:
                self._refresh_stuck_locked()
                selected = self._select_records_for_report_locked(ids)
                pending = [
                    record
                    for record in selected
                    if record.status in ACTIVE_STATUSES
                    and record.future is not None
                    and not record.future.done()
                ]
            if not pending:
                break
            if deadline is not None and time.monotonic() >= deadline:
                break
            time.sleep(0.1)

        report = self.get_subagent_report(ids)
        all_done = all(worker["status"] not in ACTIVE_STATUSES for worker in report["workers"])
        report["all_done"] = all_done
        return report

    def revise_subagent(
        self,
        subagent_id: str,
        feedback: str,
        *,
        source: str = "orchestrator",
    ) -> dict[str, Any]:
        clean_feedback = feedback.strip()
        if not clean_feedback:
            return {"ok": False, "error": "Feedback cannot be empty."}

        with self._lock:
            record = self._records.get(subagent_id)
            if record is None:
                return {"ok": False, "error": f"Unknown sub-agent id: {subagent_id}"}
            if self._closed:
                return {"ok": False, "error": "Sub-agent runtime is not available."}

            payload = f"Revision request from {source}: {clean_feedback}"
            record.pending_feedback.append(payload)
            if record.status in ACTIVE_STATUSES:
                record.status = "restarting"
                record.stop_reason = "revision_requested"
                record.latest_report = "Revision requested. Restarting with new guidance."
                record.updated_at = time.time()
                record.stop_event.set()
            else:
                self._restart_with_feedback_locked(record)

            snapshot = self._record_snapshot_locked(record)
            snapshot["ok"] = True
            return snapshot

    def interrupt_subagent(
        self,
        subagent_id: str,
        instruction: str,
        *,
        source: str = "orchestrator",
    ) -> dict[str, Any]:
        clean_instruction = instruction.strip()
        if not clean_instruction:
            return {"ok": False, "error": "Instruction cannot be empty."}
        return self.revise_subagent(
            subagent_id,
            f"Interrupt instruction from {source}: {clean_instruction}",
            source=source,
        )

    def terminate_subagent(
        self,
        subagent_id: str,
        *,
        reason: str = "",
    ) -> dict[str, Any]:
        with self._lock:
            record = self._records.get(subagent_id)
            if record is None:
                return {"ok": False, "error": f"Unknown sub-agent id: {subagent_id}"}

            if record.status in ACTIVE_STATUSES:
                record.terminate_requested = True
                record.pending_feedback.clear()
                record.status = "terminating"
                record.stop_reason = (
                    f"terminated:{reason.strip()}" if reason.strip() else "terminated"
                )
                record.latest_report = "Termination requested."
                record.updated_at = time.time()
                record.stop_event.set()
            else:
                record.terminate_requested = True
                record.status = "terminated"
                record.stop_reason = (
                    f"terminated:{reason.strip()}" if reason.strip() else "terminated"
                )
                record.latest_report = "Terminated."
                record.updated_at = time.time()

            snapshot = self._record_snapshot_locked(record)
            snapshot["ok"] = True
            return snapshot

    def _submit_locked(self, record: SubAgentRecord) -> None:
        record.stop_event.clear()
        record.stop_reason = ""
        record.error = ""
        record.status = "running"
        record.completion_report = ""
        record.latest_report = "Running."
        now = time.time()
        record.started_at = now
        record.last_activity_at = now
        record.updated_at = now
        future = self._executor.submit(self._worker_entry, record.id, record.attempt)
        record.future = future
        future.add_done_callback(
            lambda fut, sid=record.id, attempt=record.attempt: self._on_worker_done(
                sid,
                attempt,
                fut,
            )
        )

    def _worker_entry(self, subagent_id: str, attempt: int) -> dict[str, Any]:
        with self._lock:
            record = self._records.get(subagent_id)
            if record is None or record.attempt != attempt:
                return {"status": "terminated", "stop_reason": "stale_attempt"}
            task = record.task
            context = record.context
            thread_id = record.thread_id
            settings = self._settings

        agent = self._build_subagent(settings)
        return asyncio.run(
            self._run_subagent(
                agent=agent,
                subagent_id=subagent_id,
                attempt=attempt,
                task=task,
                context=context,
                thread_id=thread_id,
            )
        )

    async def _run_subagent(
        self,
        *,
        agent: Any,
        subagent_id: str,
        attempt: int,
        task: str,
        context: str,
        thread_id: str,
    ) -> dict[str, Any]:
        started_monotonic = time.monotonic()
        report_interval = float(self._settings.sub_agent_report_interval_s)
        max_runtime_s = float(self._settings.sub_agent_max_runtime_s)
        next_report_at = started_monotonic + report_interval
        tool_calls = 0
        output_chars = 0
        final_parts: list[str] = []
        fallback_text = ""

        try:
            async for mode, chunk in agent.astream(
                {"messages": [HumanMessage(content=_render_assignment(task, context))]},
                config={"configurable": {"thread_id": thread_id}},
                stream_mode=["messages", "updates"],
            ):
                self._guard_stop(subagent_id, attempt)
                elapsed = time.monotonic() - started_monotonic
                if elapsed > max_runtime_s:
                    raise _StopRequested("max_runtime_exceeded")

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
                            tool_calls += len(getattr(message, "tool_calls", None) or [])
                            maybe_text = message_text(message.content)
                            if maybe_text:
                                fallback_text = maybe_text

                now_monotonic = time.monotonic()
                if now_monotonic >= next_report_at:
                    self._update_progress(
                        subagent_id,
                        attempt,
                        tool_calls,
                        output_chars,
                        report=(
                            f"Running attempt {attempt}: "
                            f"tool_calls={tool_calls}, output_chars={output_chars}."
                        ),
                    )
                    next_report_at = now_monotonic + report_interval
                else:
                    self._update_progress(
                        subagent_id,
                        attempt,
                        tool_calls,
                        output_chars,
                    )
        except _StopRequested as exc:
            status = "stuck" if exc.reason == "max_runtime_exceeded" else "terminated"
            return {
                "status": status,
                "stop_reason": exc.reason,
                "tool_calls": tool_calls,
                "output_chars": output_chars,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "failed",
                "error": str(exc),
                "tool_calls": tool_calls,
                "output_chars": output_chars,
            }

        final_output = "".join(final_parts).strip()
        if not final_output:
            final_output = fallback_text.strip()
        if len(final_output) > _MAX_STORED_FINAL_OUTPUT_CHARS:
            final_output = (
                final_output[: _MAX_STORED_FINAL_OUTPUT_CHARS - 3] + "..."
            )
        completion_report = await _build_completion_report(
            agent=agent,
            thread_id=thread_id,
            settings=self._settings,
            goal=task,
            final_output=final_output,
        )
        return {
            "status": "completed",
            "final_output": final_output,
            "completion_report": completion_report,
            "tool_calls": tool_calls,
            "output_chars": output_chars,
        }

    def _guard_stop(self, subagent_id: str, attempt: int) -> None:
        with self._lock:
            record = self._records.get(subagent_id)
            if record is None or record.attempt != attempt:
                raise _StopRequested("stale_attempt")
            if record.stop_event.is_set():
                reason = record.stop_reason or "stop_requested"
                raise _StopRequested(reason)

    def _update_progress(
        self,
        subagent_id: str,
        attempt: int,
        tool_calls: int,
        output_chars: int,
        *,
        report: str = "",
    ) -> None:
        with self._lock:
            record = self._records.get(subagent_id)
            if record is None or record.attempt != attempt:
                return
            now = time.time()
            record.updated_at = now
            record.last_activity_at = now
            record.tool_calls = tool_calls
            record.output_chars = output_chars
            if report:
                record.latest_report = report

    def _on_worker_done(
        self,
        subagent_id: str,
        attempt: int,
        future: Future[dict[str, Any]],
    ) -> None:
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001
            result = {"status": "failed", "error": str(exc)}

        with self._lock:
            record = self._records.get(subagent_id)
            if record is None or record.attempt != attempt:
                return

            now = time.time()
            record.updated_at = now
            record.finished_at = now
            record.last_activity_at = now
            record.status = result.get("status", "failed")
            record.stop_reason = result.get("stop_reason", "")
            record.error = result.get("error", "")
            record.final_output = result.get("final_output", "")
            record.completion_report = result.get("completion_report", "")
            record.tool_calls = max(record.tool_calls, result.get("tool_calls", 0))
            record.output_chars = max(record.output_chars, result.get("output_chars", 0))

            if record.pending_feedback and not record.terminate_requested:
                self._restart_with_feedback_locked(record)
                return

            if record.status == "completed":
                record.latest_report = "Completed."
            elif record.status == "failed":
                record.latest_report = f"Failed: {record.error or 'unknown error'}"
            elif record.status == "stuck":
                record.latest_report = "Marked stuck and stopped due to runtime limit."
            else:
                record.latest_report = "Stopped."

    def _restart_with_feedback_locked(self, record: SubAgentRecord) -> None:
        feedback = "\n".join(item.strip() for item in record.pending_feedback if item.strip())
        record.pending_feedback.clear()
        record.attempt += 1
        record.task = _task_with_feedback(record.base_task, feedback, record.final_output)
        record.stop_event.clear()
        record.status = "restarting"
        record.error = ""
        record.final_output = ""
        record.completion_report = ""
        record.finished_at = 0.0
        record.latest_report = "Restarting with feedback."
        self._submit_locked(record)

    def _refresh_stuck_locked(self) -> None:
        now = time.time()
        max_runtime = float(self._settings.sub_agent_max_runtime_s)
        for record in self._records.values():
            if record.status not in ACTIVE_STATUSES:
                continue
            if record.started_at <= 0:
                continue
            if now - record.started_at <= max_runtime:
                continue
            record.status = "terminating"
            record.stop_reason = "max_runtime_exceeded"
            record.latest_report = "Runtime limit exceeded. Stopping."
            record.updated_at = now
            record.stop_event.set()

    def _record_snapshot_locked(self, record: SubAgentRecord) -> dict[str, Any]:
        goal = _preview_text(record.base_task, _MAX_TASK_PREVIEW_CHARS)
        snapshot: dict[str, Any] = {
            "id": record.id,
            "status": record.status,
            "attempt": record.attempt,
            "goal": goal,
            "task": goal,
        }

        if record.status in ACTIVE_STATUSES:
            progress = _preview_text(record.latest_report, _MAX_REPORT_PREVIEW_CHARS)
            if progress:
                snapshot["progress"] = progress
                snapshot["latest_report"] = progress
            return snapshot

        if record.status == "completed":
            final_result = (
                record.completion_report.strip()
                or record.final_output.strip()
                or "Completed."
            )
            snapshot["final_result"] = _preview_text(
                final_result,
                _MAX_FINAL_OUTPUT_PREVIEW_CHARS,
            )
            return snapshot

        if record.status == "failed":
            snapshot["error"] = _preview_text(
                record.error or "unknown error",
                _MAX_ERROR_PREVIEW_CHARS,
            )
            return snapshot

        if record.stop_reason:
            snapshot["stop_reason"] = _preview_text(
                record.stop_reason,
                _MAX_ERROR_PREVIEW_CHARS,
            )
        progress = _preview_text(record.latest_report, _MAX_REPORT_PREVIEW_CHARS)
        if progress:
            snapshot["progress"] = progress
            snapshot["latest_report"] = progress
        return snapshot

    def _active_count_locked(self) -> int:
        return sum(1 for record in self._records.values() if record.status in ACTIVE_STATUSES)

    def _select_records_locked(self, ids: list[str] | None) -> list[SubAgentRecord]:
        if not ids:
            return list(self._records.values())
        selected: list[SubAgentRecord] = []
        for subagent_id in ids:
            record = self._records.get(subagent_id)
            if record is not None:
                selected.append(record)
        return selected

    def _select_records_for_report_locked(
        self,
        ids: list[str] | None,
    ) -> list[SubAgentRecord]:
        selected = self._select_records_locked(ids)
        if ids is not None:
            return selected

        active = [record for record in selected if record.status in ACTIVE_STATUSES]
        inactive = [record for record in selected if record.status not in ACTIVE_STATUSES]
        inactive.sort(key=lambda record: record.updated_at, reverse=True)
        keep = max(_MAX_REPORT_WORKERS - len(active), 0)
        trimmed = [*active, *inactive[:keep]]
        trimmed.sort(key=lambda record: record.created_at)
        return trimmed


def _render_assignment(task: str, context: str) -> str:
    if not context.strip():
        return (
            "Assigned task:\n"
            f"{task.strip()}\n\n"
            "Execute this task and provide a clear final result."
        )
    return (
        "Assigned task:\n"
        f"{task.strip()}\n\n"
        "Relevant context:\n"
        f"{context.strip()}\n\n"
        "Execute the task using the context and provide a clear final result."
    )


def _task_with_feedback(base_task: str, feedback: str, prior_output: str) -> str:
    if not feedback and not prior_output:
        return base_task
    parts = [f"Original task:\n{base_task.strip()}"]
    if prior_output.strip():
        trimmed_prior = _preview_text(prior_output.strip(), _MAX_REVISION_PART_CHARS)
        parts.append(f"Prior attempt output:\n{trimmed_prior}")
    if feedback.strip():
        trimmed_feedback = _preview_text(feedback.strip(), _MAX_REVISION_PART_CHARS)
        parts.append(f"Revision instructions:\n{trimmed_feedback}")
    parts.append("Produce an improved final result.")
    return "\n\n".join(parts)


async def _build_completion_report(
    *,
    agent: Any,
    thread_id: str,
    settings: Settings,
    goal: str,
    final_output: str,
) -> str:
    fallback = _preview_text(final_output.strip(), _MAX_COMPLETION_REPORT_CHARS)
    if not fallback:
        fallback = "Task completed."
    try:
        messages = await _aget_worker_messages(agent, thread_id)
        transcript = _render_history_for_report(messages)
        if not transcript:
            return fallback

        model = build_chat_model(settings, settings.compact_model_name)
        response = await asyncio.wait_for(
            model.ainvoke(
                [
                    SystemMessage(content=_SUBAGENT_REPORT_PROMPT),
                    HumanMessage(
                        content=(
                            "Goal:\n"
                            f"{_preview_text(goal.strip(), _MAX_TASK_PREVIEW_CHARS)}\n\n"
                            "Final output:\n"
                            f"{_preview_text(final_output.strip(), _MAX_REPORT_INPUT_FINAL_CHARS)}\n\n"
                            "Worker transcript:\n"
                            f"{transcript}"
                        )
                    ),
                ]
            ),
            timeout=_REPORT_LLM_TIMEOUT_S,
        )
        report = message_text(response.content, include_reasoning=False).strip()
        if not report:
            return fallback
        return _preview_text(report, _MAX_COMPLETION_REPORT_CHARS)
    except Exception:  # noqa: BLE001
        return fallback


async def _aget_worker_messages(agent: Any, thread_id: str) -> list[Any]:
    state = await agent.aget_state({"configurable": {"thread_id": thread_id}})
    values = getattr(state, "values", {})
    messages = values.get("messages", [])
    if not isinstance(messages, list):
        return []
    return list(messages)


def _render_history_for_report(messages: list[Any]) -> str:
    if not messages:
        return ""

    remaining = _MAX_REPORT_HISTORY_CHARS
    rendered: list[str] = []
    for idx, message in enumerate(messages, start=1):
        heading = _message_heading(message)
        body = _message_body(message)
        if not body:
            continue
        body = _preview_text(body, _MAX_HISTORY_MESSAGE_CHARS)
        block = f"[{idx}] {heading}\n{body}\n\n"
        if len(block) <= remaining:
            rendered.append(block)
            remaining -= len(block)
            continue
        if remaining <= 40:
            break
        rendered.append(block[: remaining - 20] + "\n...[truncated]\n")
        break
    return "".join(rendered).strip()


def _message_heading(message: Any) -> str:
    if isinstance(message, HumanMessage):
        return "User"
    if isinstance(message, ToolMessage):
        return "Tool"
    if isinstance(message, SystemMessage):
        return "System"
    if isinstance(message, AIMessage):
        return "Assistant"
    return "Message"


def _message_body(message: Any) -> str:
    content = message_text(getattr(message, "content", ""), include_reasoning=False).strip()
    if content:
        return content
    if isinstance(message, AIMessage):
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            names = ", ".join(str(tc.get("name", "tool")) for tc in tool_calls[:6])
            return f"Tool calls: {names}"
    return ""


def _preview_text(text: str, limit: int) -> str:
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."

