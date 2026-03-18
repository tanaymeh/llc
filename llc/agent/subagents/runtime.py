from __future__ import annotations

import asyncio
import random
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
from llc.service.prompt_registry import PromptRegistry

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
_MAX_ACTIVITY_PREVIEW_CHARS = 80
_VICTORIAN_NAMES: tuple[str, ...] = (
    "Mr Darcy",
    "Miss Bennet",
    "Lady Bronte",
    "Lord Tennyson",
    "Mrs Gaskell",
    "Sir Fairfax",
    "Miss Elinor",
    "Master Pip",
    "Lady Ada",
    "Mr Bingley",
    "Miss Nightingale",
    "Lord Ashford",
    "Mrs Templeton",
    "Sir Whitmore",
    "Miss Hawthorne",
    "Mr Pembroke",
    "Lady Winthrop",
    "Captain Blackwood",
    "Miss Worthing",
    "Reverend Pritchard",
    "Baroness Carlisle",
    "Mr Thackeray",
    "Lady Beatrice",
    "Miss Amelia",
    "Lord Rochester",
)
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
    def __init__(
        self,
        settings: Settings,
        build_subagent: SubAgentBuilder,
        prompt_registry: PromptRegistry | None = None,
    ) -> None:
        self._settings = settings
        self._build_subagent = build_subagent
        self._prompt_registry = prompt_registry
        self._records: dict[str, SubAgentRecord] = {}
        self._lock = RLock()
        self._closed = False
        self._usage_input_tokens = 0
        self._usage_output_tokens = 0
        self._usage_by_model: dict[str, dict[str, int]] = {}
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

            provided_name = (name or "").strip()
            existing_names = {
                str(existing.name).strip().lower()
                for existing in self._records.values()
                if str(existing.name).strip()
            }
            selected_name = provided_name or _pick_victorian_name(existing_names)
            subagent_id = _build_subagent_id(selected_name)
            while subagent_id in self._records:
                subagent_id = _build_subagent_id(selected_name)
            now = time.time()
            record = SubAgentRecord(
                id=subagent_id,
                name=selected_name,
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

    def get_subagent_report(
        self,
        ids: list[str] | None = None,
        *,
        include_all: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            self._refresh_stuck_locked()
            selected = self._select_records_for_report_locked(
                ids,
                include_all=include_all,
            )
            workers = [self._record_snapshot_locked(record) for record in selected]
            return {
                "ok": True,
                "active_count": self._active_count_locked(),
                "max_sub_agents": self._settings.max_sub_agents,
                "workers": workers,
            }

    def get_usage_totals(self) -> dict[str, Any]:
        with self._lock:
            by_model = {
                model_name: {
                    "input_tokens": int(bucket.get("input_tokens", 0) or 0),
                    "output_tokens": int(bucket.get("output_tokens", 0) or 0),
                }
                for model_name, bucket in self._usage_by_model.items()
            }
            return {
                "input_tokens": self._usage_input_tokens,
                "output_tokens": self._usage_output_tokens,
                "by_model": by_model,
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
                record.current_activity = "Restarting with feedback"
                record.activity_detail = _preview_text(
                    clean_feedback,
                    _MAX_REPORT_PREVIEW_CHARS,
                )
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
                record.current_activity = "Terminating"
                if reason.strip():
                    record.activity_detail = _preview_text(
                        reason.strip(),
                        _MAX_REPORT_PREVIEW_CHARS,
                    )
                record.updated_at = time.time()
                record.stop_event.set()
            else:
                record.terminate_requested = True
                record.status = "terminated"
                record.stop_reason = (
                    f"terminated:{reason.strip()}" if reason.strip() else "terminated"
                )
                record.latest_report = "Terminated."
                record.current_activity = "Agent de-spawned"
                record.activity_detail = "Terminated."
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
        record.current_activity = "Running"
        record.activity_detail = "Worker started."
        record.last_tool_name = ""
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
                settings=settings,
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
        settings: Settings,
    ) -> dict[str, Any]:
        started_monotonic = time.monotonic()
        report_interval = float(settings.sub_agent_report_interval_s)
        max_runtime_s = float(settings.sub_agent_max_runtime_s)
        next_report_at = started_monotonic + report_interval
        tool_calls = 0
        output_chars = 0
        final_parts: list[str] = []
        fallback_text = ""
        usage_by_model: dict[str, dict[str, int]] = {}
        last_tool_name = ""
        current_activity = "Running"
        activity_detail = "Worker started."

        try:
            async for mode, chunk in agent.astream(
                {
                    "messages": [
                        HumanMessage(
                            content=_render_assignment(
                                task,
                                context,
                                self._prompt_registry,
                            )
                        )
                    ]
                },
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
                            tool_calls_batch = getattr(message, "tool_calls", None) or []
                            tool_calls += len(tool_calls_batch)
                            if tool_calls_batch:
                                raw_tool_name = tool_calls_batch[-1].get("name", "")
                                if isinstance(raw_tool_name, str) and raw_tool_name.strip():
                                    last_tool_name = raw_tool_name.strip()
                                    current_activity = _activity_from_tool_name(last_tool_name)
                                    activity_detail = f"Using {last_tool_name}."
                            usage_input, usage_output = _token_usage(message)
                            _accumulate_usage_by_model(
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
                                    activity_detail = _preview_text(
                                        maybe_text,
                                        _MAX_REPORT_PREVIEW_CHARS,
                                    )

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
                        activity=current_activity,
                        activity_detail=activity_detail,
                        last_tool_name=last_tool_name,
                    )
                    next_report_at = now_monotonic + report_interval
                else:
                    self._update_progress(
                        subagent_id,
                        attempt,
                        tool_calls,
                        output_chars,
                        activity=current_activity,
                        activity_detail=activity_detail,
                        last_tool_name=last_tool_name,
                    )
        except _StopRequested as exc:
            status = "stuck" if exc.reason == "max_runtime_exceeded" else "terminated"
            return {
                "status": status,
                "stop_reason": exc.reason,
                "tool_calls": tool_calls,
                "output_chars": output_chars,
                "usage_by_model": usage_by_model,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "failed",
                "error": str(exc),
                "tool_calls": tool_calls,
                "output_chars": output_chars,
                "usage_by_model": usage_by_model,
            }

        final_output = "".join(final_parts).strip()
        if not final_output:
            final_output = fallback_text.strip()
        if len(final_output) > _MAX_STORED_FINAL_OUTPUT_CHARS:
            final_output = (
                final_output[: _MAX_STORED_FINAL_OUTPUT_CHARS - 3] + "..."
            )
        self._update_progress(
            subagent_id,
            attempt,
            tool_calls,
            output_chars,
            report="Generating completion report.",
            activity="Generating report",
            activity_detail="Summarizing sub-agent output.",
            last_tool_name=last_tool_name,
        )
        report_prompt = self._completion_report_prompt()
        (
            completion_report,
            report_input_tokens,
            report_output_tokens,
            report_model_name,
        ) = await _build_completion_report(
            agent=agent,
            thread_id=thread_id,
            settings=settings,
            goal=task,
            final_output=final_output,
            report_prompt=report_prompt,
        )
        _accumulate_usage_by_model(
            usage_by_model,
            report_model_name,
            report_input_tokens,
            report_output_tokens,
        )
        return {
            "status": "completed",
            "final_output": final_output,
            "completion_report": completion_report,
            "tool_calls": tool_calls,
            "output_chars": output_chars,
            "usage_by_model": usage_by_model,
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
        activity: str = "",
        activity_detail: str = "",
        last_tool_name: str = "",
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
            if activity:
                record.current_activity = _preview_text(
                    activity,
                    _MAX_ACTIVITY_PREVIEW_CHARS,
                )
            if activity_detail:
                record.activity_detail = _preview_text(
                    activity_detail,
                    _MAX_REPORT_PREVIEW_CHARS,
                )
            if last_tool_name:
                record.last_tool_name = _preview_text(
                    last_tool_name,
                    _MAX_ACTIVITY_PREVIEW_CHARS,
                )

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
            self._merge_usage_locked(_normalize_usage_by_model(result.get("usage_by_model")))

            if record.pending_feedback and not record.terminate_requested:
                self._restart_with_feedback_locked(record)
                return

            if record.status == "completed":
                record.latest_report = "Completed."
                record.activity_detail = "Completed successfully."
            elif record.status == "failed":
                record.latest_report = f"Failed: {record.error or 'unknown error'}"
                record.activity_detail = _preview_text(
                    record.error or "unknown error",
                    _MAX_REPORT_PREVIEW_CHARS,
                )
            elif record.status == "stuck":
                record.latest_report = "Marked stuck and stopped due to runtime limit."
                record.activity_detail = "Stopped after runtime limit."
            else:
                record.latest_report = "Stopped."
                record.activity_detail = "Stopped before completion."
            record.current_activity = "Agent de-spawned"

    def _restart_with_feedback_locked(self, record: SubAgentRecord) -> None:
        feedback = "\n".join(item.strip() for item in record.pending_feedback if item.strip())
        record.pending_feedback.clear()
        record.attempt += 1
        record.task = _task_with_feedback(
            record.base_task,
            feedback,
            record.final_output,
            self._prompt_registry,
        )
        record.stop_event.clear()
        record.status = "restarting"
        record.error = ""
        record.final_output = ""
        record.completion_report = ""
        record.finished_at = 0.0
        record.latest_report = "Restarting with feedback."
        record.current_activity = "Restarting with feedback"
        if feedback:
            record.activity_detail = _preview_text(feedback, _MAX_REPORT_PREVIEW_CHARS)
        record.last_tool_name = ""
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
            record.current_activity = "Terminating"
            record.activity_detail = "Runtime limit exceeded."
            record.updated_at = now
            record.stop_event.set()

    def _record_snapshot_locked(self, record: SubAgentRecord) -> dict[str, Any]:
        goal = _preview_text(record.base_task, _MAX_TASK_PREVIEW_CHARS)
        current_task = _preview_text(record.task, _MAX_TASK_PREVIEW_CHARS)
        if record.status in ACTIVE_STATUSES:
            current_activity = record.current_activity or "Running"
        else:
            current_activity = "Agent de-spawned"
        activity_detail = record.activity_detail or record.latest_report
        snapshot: dict[str, Any] = {
            "id": record.id,
            "status": record.status,
            "attempt": record.attempt,
            "name": record.name,
            "source": record.source,
            "goal": goal,
            "task": goal,
            "current_task": current_task,
            "current_activity": _preview_text(
                current_activity,
                _MAX_ACTIVITY_PREVIEW_CHARS,
            ),
            "activity_detail": _preview_text(
                activity_detail,
                _MAX_REPORT_PREVIEW_CHARS,
            ),
            "last_tool_name": _preview_text(
                record.last_tool_name,
                _MAX_ACTIVITY_PREVIEW_CHARS,
            ),
            "tool_calls": max(int(record.tool_calls or 0), 0),
            "output_chars": max(int(record.output_chars or 0), 0),
            "created_at": float(record.created_at or 0),
            "updated_at": float(record.updated_at or 0),
            "started_at": float(record.started_at or 0),
            "finished_at": float(record.finished_at or 0),
        }

        latest_report = _preview_text(record.latest_report, _MAX_REPORT_PREVIEW_CHARS)
        if latest_report:
            snapshot["latest_report"] = latest_report

        if record.status in ACTIVE_STATUSES:
            if latest_report:
                snapshot["progress"] = latest_report
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
        if latest_report:
            snapshot["progress"] = latest_report
        return snapshot

    def _active_count_locked(self) -> int:
        return sum(1 for record in self._records.values() if record.status in ACTIVE_STATUSES)

    def _merge_usage_locked(self, usage_by_model: dict[str, dict[str, int]]) -> None:
        for model_name, usage in usage_by_model.items():
            input_tokens = max(int(usage.get("input_tokens", 0) or 0), 0)
            output_tokens = max(int(usage.get("output_tokens", 0) or 0), 0)
            if input_tokens == 0 and output_tokens == 0:
                continue
            bucket = self._usage_by_model.setdefault(
                model_name,
                {"input_tokens": 0, "output_tokens": 0},
            )
            bucket["input_tokens"] += input_tokens
            bucket["output_tokens"] += output_tokens
            self._usage_input_tokens += input_tokens
            self._usage_output_tokens += output_tokens

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
        *,
        include_all: bool = False,
    ) -> list[SubAgentRecord]:
        selected = self._select_records_locked(ids)
        if ids is not None:
            return selected
        if include_all:
            selected.sort(key=lambda record: record.created_at)
            return selected

        active = [record for record in selected if record.status in ACTIVE_STATUSES]
        inactive = [record for record in selected if record.status not in ACTIVE_STATUSES]
        inactive.sort(key=lambda record: record.updated_at, reverse=True)
        keep = max(_MAX_REPORT_WORKERS - len(active), 0)
        trimmed = [*active, *inactive[:keep]]
        trimmed.sort(key=lambda record: record.created_at)
        return trimmed

    def _completion_report_prompt(self) -> str:
        registry = self._prompt_registry
        if registry is not None:
            try:
                return registry.get(
                    "subagent_report",
                    key="subagent_report_prompt",
                )
            except Exception:
                return _SUBAGENT_REPORT_PROMPT
        return _SUBAGENT_REPORT_PROMPT


def _render_assignment(
    task: str,
    context: str,
    prompt_registry: PromptRegistry | None = None,
) -> str:
    clean_task = task.strip()
    clean_context = context.strip()
    if prompt_registry is not None:
        if clean_context:
            try:
                return prompt_registry.get_formatted(
                    "assignment",
                    key="assignment_with_context_prompt",
                    task=clean_task,
                    context=clean_context,
                )
            except Exception:
                pass
        try:
            return prompt_registry.get_formatted(
                "assignment",
                key="assignment_prompt",
                task=clean_task,
            )
        except Exception:
            pass
    if not clean_context:
        return (
            "Assigned task:\n"
            f"{clean_task}\n\n"
            "Execute this task and provide a clear final result."
        )
    return (
        "Assigned task:\n"
        f"{clean_task}\n\n"
        "Relevant context:\n"
        f"{clean_context}\n\n"
        "Execute the task using the context and provide a clear final result."
    )


def _task_with_feedback(
    base_task: str,
    feedback: str,
    prior_output: str,
    prompt_registry: PromptRegistry | None = None,
) -> str:
    base = base_task.strip()
    prior = prior_output.strip()
    revised = feedback.strip()
    trimmed_prior = _preview_text(prior, _MAX_REVISION_PART_CHARS) if prior else ""
    trimmed_feedback = _preview_text(revised, _MAX_REVISION_PART_CHARS) if revised else ""

    if prompt_registry is not None:
        try:
            rendered = prompt_registry.get_formatted(
                "assignment",
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


async def _build_completion_report(
    *,
    agent: Any,
    thread_id: str,
    settings: Settings,
    goal: str,
    final_output: str,
    report_prompt: str,
) -> tuple[str, int, int, str]:
    report_model_name = settings.compact_model_name or settings.model_name
    fallback = _preview_text(final_output.strip(), _MAX_COMPLETION_REPORT_CHARS)
    if not fallback:
        fallback = "Task completed."
    try:
        messages = await _aget_worker_messages(agent, thread_id)
        transcript = _render_history_for_report(messages)
        if not transcript:
            return fallback, 0, 0, report_model_name

        model = build_chat_model(settings, settings.compact_model_name)
        response = await asyncio.wait_for(
            model.ainvoke(
                [
                    SystemMessage(content=report_prompt),
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
        usage_input, usage_output = _token_usage(response)
        report = message_text(response.content, include_reasoning=False).strip()
        if not report:
            return fallback, usage_input, usage_output, report_model_name
        return (
            _preview_text(report, _MAX_COMPLETION_REPORT_CHARS),
            usage_input,
            usage_output,
            report_model_name,
        )
    except Exception:  # noqa: BLE001
        return fallback, 0, 0, report_model_name


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


def _activity_from_tool_name(tool_name: str) -> str:
    normalized = tool_name.strip().lower()
    if not normalized:
        return "Working"
    if normalized in {"edit", "multiedit", "write"}:
        return "Writing code"
    if normalized in {"websearch", "webfetch"}:
        return "Searching web"
    if normalized in {"grep", "glob", "code_grep", "read", "ls"}:
        return "Searching codebase"
    if normalized == "bash":
        return "Running shell commands"
    if normalized == "showdiff":
        return "Reviewing diffs"
    if normalized == "todowrite":
        return "Updating plan"
    return "Working"


def _pick_victorian_name(used_names: set[str]) -> str:
    available = [name for name in _VICTORIAN_NAMES if name.lower() not in used_names]
    if available:
        return random.choice(available)
    return random.choice(_VICTORIAN_NAMES)


def _slugify_label(value: str) -> str:
    parts: list[str] = []
    for ch in value.lower():
        if ch.isalnum():
            parts.append(ch)
        elif ch in {" ", "-", "_"}:
            parts.append("-")
    slug = "".join(parts)
    slug = "-".join(part for part in slug.split("-") if part)
    if not slug:
        return "victorian-agent"
    if len(slug) <= 24:
        return slug
    return slug[:24].rstrip("-")


def _build_subagent_id(name: str) -> str:
    slug = _slugify_label(name)
    return f"subagent-{slug}-{uuid.uuid4().hex[:4]}"


def _token_usage(message: Any) -> tuple[int, int]:
    usage = getattr(message, "usage_metadata", None)
    if not usage:
        return 0, 0
    if isinstance(usage, dict):
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        return max(input_tokens, 0), max(output_tokens, 0)
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    return max(input_tokens, 0), max(output_tokens, 0)


def _accumulate_usage_by_model(
    usage_by_model: dict[str, dict[str, int]],
    model_name: str,
    input_tokens: int,
    output_tokens: int,
) -> None:
    in_tokens = max(int(input_tokens or 0), 0)
    out_tokens = max(int(output_tokens or 0), 0)
    if not model_name or (in_tokens == 0 and out_tokens == 0):
        return
    bucket = usage_by_model.setdefault(
        model_name,
        {"input_tokens": 0, "output_tokens": 0},
    )
    bucket["input_tokens"] += in_tokens
    bucket["output_tokens"] += out_tokens


def _normalize_usage_by_model(raw_usage: Any) -> dict[str, dict[str, int]]:
    if not isinstance(raw_usage, dict):
        return {}
    normalized: dict[str, dict[str, int]] = {}
    for raw_model_name, raw_bucket in raw_usage.items():
        model_name = str(raw_model_name).strip()
        if not model_name or not isinstance(raw_bucket, dict):
            continue
        input_tokens = max(int(raw_bucket.get("input_tokens", 0) or 0), 0)
        output_tokens = max(int(raw_bucket.get("output_tokens", 0) or 0), 0)
        if input_tokens == 0 and output_tokens == 0:
            continue
        normalized[model_name] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    return normalized

