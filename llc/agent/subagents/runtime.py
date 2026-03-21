from __future__ import annotations

import asyncio
import multiprocessing as mp
import random
import sys
import time
import uuid
from queue import Empty
from threading import RLock
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from llc.agent.message_utils import message_text
from llc.agent.subagents.types import ACTIVE_STATUSES, SubAgentRecord
from llc.config import Settings
from llc.observability import (
    capture_parent_context,
    merge_langchain_config,
    start_linked_subagent_trace,
    update_observation,
)
from llc.service.prompt_registry import PromptRegistry

SubAgentBuilder = Callable[[Settings], Any]
_MAX_STORED_FINAL_OUTPUT_CHARS = 12000
_MAX_TASK_PREVIEW_CHARS = 180
_MAX_REPORT_PREVIEW_CHARS = 220
_MAX_ERROR_PREVIEW_CHARS = 220
_MAX_FINAL_OUTPUT_PREVIEW_CHARS = 600
_MAX_REVISION_PART_CHARS = 1800
_MAX_REPORT_WORKERS = 8
_MAX_ACTIVITY_PREVIEW_CHARS = 80
_WAIT_POLL_INTERVAL_S = 0.1
_STUCK_STOP_REASONS = {
    "max_runtime_exceeded",
    "stall_timeout_exceeded",
    "max_tool_calls_exceeded",
}
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
        main_file = getattr(sys.modules.get("__main__"), "__file__", "") or ""
        prefer_spawn = bool(main_file and not str(main_file).startswith("<"))
        if prefer_spawn:
            self._mp_context = mp.get_context("spawn")
        else:
            try:
                self._mp_context = mp.get_context("fork")
            except ValueError:
                self._mp_context = mp.get_context("spawn")

    def update_settings(self, settings: Settings) -> None:
        with self._lock:
            self._settings = settings

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            now = time.time()
            for record in self._records.values():
                if record.status in ACTIVE_STATUSES or self._is_worker_alive_locked(record):
                    record.terminate_requested = True
                    record.stop_reason = "runtime_shutdown"
                    record.status = "terminating"
                    record.stop_requested_at = now
                    self._request_worker_stop_locked(record)
                    self._terminate_worker_process_locked(record)
                    self._cleanup_worker_handles_locked(record)

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
            active_count = self._active_count_locked()
            capacity_count = self._capacity_count_locked()
            if capacity_count >= self._settings.max_sub_agents:
                capacity_suffix = ""
                if active_count < capacity_count:
                    capacity_suffix = (
                        " Some workers are unresponsive and still consuming runner capacity."
                    )
                return {
                    "ok": False,
                    "error": (
                        f"Maximum active sub-agents reached "
                        f"({self._settings.max_sub_agents}).{capacity_suffix}"
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
            parent_context = capture_parent_context() or {}
            now = time.time()
            record = SubAgentRecord(
                id=subagent_id,
                name=selected_name,
                base_task=clean_task,
                task=clean_task,
                context=context.strip(),
                source=source,
                thread_id=f"{subagent_id}-{uuid.uuid4().hex[:6]}",
                parent_trace_id=str(parent_context.get("trace_id", "")).strip(),
                parent_observation_id=str(
                    parent_context.get("parent_observation_id", "")
                ).strip(),
                parent_session_id=str(parent_context.get("session_id", "")).strip(),
                parent_turn_id=str(parent_context.get("turn_id", "")).strip(),
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
        effective_timeout_ms = 0
        if timeout_ms is None:
            effective_timeout_ms = int(self._settings.sub_agent_wait_timeout_ms)
        elif timeout_ms > 0:
            effective_timeout_ms = int(timeout_ms)

        timeout_s = 0.0
        if effective_timeout_ms > 0:
            timeout_s = effective_timeout_ms / 1000
        deadline = time.monotonic() + timeout_s if timeout_s > 0 else None

        while True:
            with self._lock:
                self._refresh_stuck_locked()
                selected = self._select_records_for_report_locked(ids)
                pending = [
                    record
                    for record in selected
                    if record.status in ACTIVE_STATUSES
                    and self._is_worker_alive_locked(record)
                ]
            if not pending:
                break
            if deadline is not None and time.monotonic() >= deadline:
                break
            time.sleep(_WAIT_POLL_INTERVAL_S)

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
                record.stop_requested_at = record.updated_at
                self._request_worker_stop_locked(record)
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
                record.stop_requested_at = record.updated_at
                self._request_worker_stop_locked(record)
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
                self._terminate_worker_process_locked(record)
                self._cleanup_worker_handles_locked(record)

            snapshot = self._record_snapshot_locked(record)
            snapshot["ok"] = True
            return snapshot

    def _submit_locked(self, record: SubAgentRecord) -> None:
        record.stop_event = self._mp_context.Event()
        record.stop_reason = ""
        record.stop_requested_at = 0.0
        record.error = ""
        record.status = "running"
        record.latest_report = "Running."
        record.current_activity = "Running"
        record.activity_detail = "Worker started."
        record.last_tool_name = ""
        now = time.time()
        record.started_at = now
        record.last_activity_at = now
        record.updated_at = now
        record.worker_result_received = False
        worker_queue = self._mp_context.Queue()
        payload = {
            "settings": self._settings.model_dump(mode="python"),
            "subagent_id": record.id,
            "attempt": record.attempt,
            "task": record.task,
            "context": record.context,
            "thread_id": record.thread_id,
            "subagent_name": record.name,
            "parent_context": (
                {
                    "trace_id": record.parent_trace_id,
                    "parent_observation_id": record.parent_observation_id,
                    "session_id": record.parent_session_id,
                    "turn_id": record.parent_turn_id,
                }
                if record.parent_trace_id and record.parent_observation_id
                else None
            ),
        }
        process = self._mp_context.Process(
            target=_subagent_process_entry,
            kwargs={
                "payload": payload,
                "stop_event": record.stop_event,
                "worker_queue": worker_queue,
            },
            name=f"llc-subagent-{record.id}",
            daemon=False,
        )
        try:
            process.start()
        except Exception as exc:  # noqa: BLE001
            record.status = "failed"
            record.error = str(exc)
            record.latest_report = f"Failed: {record.error or 'unknown error'}"
            record.activity_detail = _preview_text(
                record.error or "unknown error",
                _MAX_REPORT_PREVIEW_CHARS,
            )
            record.current_activity = "Agent de-spawned"
            record.finished_at = time.time()
            self._close_worker_queue_safely(worker_queue)
            return
        record.worker_process = process
        record.worker_queue = worker_queue

    def _drain_worker_events_locked(self, record: SubAgentRecord) -> None:
        queue = record.worker_queue
        if queue is None:
            return
        while True:
            try:
                event = queue.get_nowait()
            except Empty:
                break
            except Exception:
                break
            if not isinstance(event, dict):
                continue
            try:
                event_attempt = int(event.get("attempt", -1) or -1)
            except Exception:
                event_attempt = -1
            if event_attempt != record.attempt:
                continue
            self._handle_worker_event_locked(record, event)
            if record.status not in ACTIVE_STATUSES:
                break

    def _handle_worker_event_locked(self, record: SubAgentRecord, event: dict[str, Any]) -> None:
        event_type = str(event.get("type", "")).strip()
        now = time.time()
        if event_type == "progress":
            tool_calls = max(int(event.get("tool_calls", 0) or 0), 0)
            output_chars = max(int(event.get("output_chars", 0) or 0), 0)
            report = str(event.get("report", "") or "")
            activity = str(event.get("activity", "") or "")
            activity_detail = str(event.get("activity_detail", "") or "")
            last_tool_name = str(event.get("last_tool_name", "") or "")
            record.updated_at = now
            record.last_activity_at = now
            record.tool_calls = max(record.tool_calls, tool_calls)
            record.output_chars = max(record.output_chars, output_chars)
            if report:
                record.latest_report = _preview_text(report, _MAX_REPORT_PREVIEW_CHARS)
            if activity:
                record.current_activity = _preview_text(activity, _MAX_ACTIVITY_PREVIEW_CHARS)
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
            return
        if event_type == "result":
            payload = event.get("result")
            result = payload if isinstance(payload, dict) else {}
            record.worker_result_received = True
            self._apply_worker_result_locked(record, result)

    def _apply_worker_result_locked(self, record: SubAgentRecord, result: dict[str, Any]) -> None:
        now = time.time()
        result_status = str(result.get("status", "failed") or "failed")
        stop_reason = str(result.get("stop_reason", "") or "")
        if stop_reason == "stop_requested" and record.stop_reason:
            stop_reason = record.stop_reason
        record.updated_at = now
        record.finished_at = now
        record.last_activity_at = now
        record.status = result_status
        record.stop_reason = stop_reason
        record.stop_requested_at = 0.0
        record.error = str(result.get("error", "") or "")
        record.final_output = str(result.get("final_output", "") or "")
        record.tool_calls = max(record.tool_calls, int(result.get("tool_calls", 0) or 0))
        record.output_chars = max(
            record.output_chars,
            int(result.get("output_chars", 0) or 0),
        )
        self._merge_usage_locked(_normalize_usage_by_model(result.get("usage_by_model")))
        self._cleanup_worker_handles_locked(record)

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
            record.latest_report = _stuck_latest_report(record.stop_reason)
            record.activity_detail = _stuck_activity_detail(record.stop_reason)
        else:
            record.latest_report = "Stopped."
            record.activity_detail = "Stopped before completion."
        record.current_activity = "Agent de-spawned"

    def _request_worker_stop_locked(self, record: SubAgentRecord) -> None:
        stop_event = record.stop_event
        if stop_event is None:
            return
        try:
            stop_event.set()
        except Exception:
            pass

    def _worker_stop_requested_locked(self, record: SubAgentRecord) -> bool:
        stop_event = record.stop_event
        if stop_event is None:
            return False
        try:
            return bool(stop_event.is_set())
        except Exception:
            return False

    def _is_worker_alive_locked(self, record: SubAgentRecord) -> bool:
        process = record.worker_process
        if process is None:
            return False
        try:
            return bool(process.is_alive())
        except Exception:
            return False

    def _terminate_worker_process_locked(self, record: SubAgentRecord) -> None:
        process = record.worker_process
        if process is None:
            return
        try:
            alive = process.is_alive()
        except Exception:
            alive = False
        if not alive:
            try:
                process.join(timeout=0)
            except Exception:
                pass
            return
        try:
            process.terminate()
        except Exception:
            pass
        try:
            process.join(timeout=0.2)
        except Exception:
            pass
        try:
            still_alive = process.is_alive()
        except Exception:
            still_alive = False
        if still_alive:
            try:
                process.kill()
            except Exception:
                pass
            try:
                process.join(timeout=0.2)
            except Exception:
                pass

    def _close_worker_queue_safely(self, queue: Any) -> None:
        try:
            queue.close()
        except Exception:
            pass
        try:
            queue.cancel_join_thread()
        except Exception:
            pass

    def _cleanup_worker_handles_locked(self, record: SubAgentRecord) -> None:
        process = record.worker_process
        if process is not None:
            try:
                process.join(timeout=0)
            except Exception:
                pass
        queue = record.worker_queue
        if queue is not None:
            self._close_worker_queue_safely(queue)
        record.worker_process = None
        record.worker_queue = None
        record.stop_event = None
        record.worker_result_received = False

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
        record.status = "restarting"
        record.error = ""
        record.final_output = ""
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
        stall_timeout = float(self._settings.sub_agent_stall_timeout_s)
        stop_grace = float(self._settings.sub_agent_stop_grace_s)
        for record in self._records.values():
            self._drain_worker_events_locked(record)
            process_alive = self._is_worker_alive_locked(record)
            if (
                not process_alive
                and record.status in ACTIVE_STATUSES
                and not record.worker_result_received
            ):
                if record.stop_reason:
                    fallback_stop_reason = record.stop_reason
                    fallback_status = (
                        "stuck" if _stop_reason_is_stuck(fallback_stop_reason) else "terminated"
                    )
                    fallback = {
                        "status": fallback_status,
                        "stop_reason": fallback_stop_reason,
                        "tool_calls": record.tool_calls,
                        "output_chars": record.output_chars,
                        "usage_by_model": {},
                    }
                else:
                    fallback = {
                        "status": "failed",
                        "error": "Worker process exited without a result payload.",
                        "tool_calls": record.tool_calls,
                        "output_chars": record.output_chars,
                        "usage_by_model": {},
                    }
                self._apply_worker_result_locked(record, fallback)
                continue
            if record.status not in ACTIVE_STATUSES:
                continue
            if record.started_at <= 0:
                continue
            last_activity_at = (
                record.last_activity_at
                if record.last_activity_at > 0
                else record.started_at
            )
            elapsed = now - record.started_at
            idle_for = now - last_activity_at
            stop_reason = ""
            if elapsed > max_runtime:
                stop_reason = "max_runtime_exceeded"
            elif idle_for > stall_timeout:
                stop_reason = "stall_timeout_exceeded"

            if stop_reason and not self._worker_stop_requested_locked(record):
                record.status = "terminating"
                record.stop_reason = stop_reason
                record.latest_report = _stuck_latest_report(stop_reason)
                record.current_activity = "Terminating"
                record.activity_detail = _stuck_activity_detail(stop_reason)
                record.updated_at = now
                record.stop_requested_at = now
                self._request_worker_stop_locked(record)
                continue

            if (
                not self._worker_stop_requested_locked(record)
                or not process_alive
                or record.stop_requested_at <= 0
                or (now - record.stop_requested_at) <= stop_grace
            ):
                continue

            base_reason = (record.stop_reason or "stop_requested").strip()
            self._terminate_worker_process_locked(record)
            if base_reason == "revision_requested":
                forced = {
                    "status": "terminated",
                    "stop_reason": "revision_requested",
                    "tool_calls": record.tool_calls,
                    "output_chars": record.output_chars,
                    "usage_by_model": {},
                }
            elif base_reason.startswith("terminated"):
                forced = {
                    "status": "terminated",
                    "stop_reason": base_reason,
                    "tool_calls": record.tool_calls,
                    "output_chars": record.output_chars,
                    "usage_by_model": {},
                }
            else:
                forced = {
                    "status": "stuck",
                    "stop_reason": f"{base_reason}:unresponsive_after_stop",
                    "tool_calls": record.tool_calls,
                    "output_chars": record.output_chars,
                    "usage_by_model": {},
                }
            self._apply_worker_result_locked(record, forced)

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
            final_result = record.final_output.strip() or "Completed."
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

    def _capacity_count_locked(self) -> int:
        count = 0
        for record in self._records.values():
            if record.status in ACTIVE_STATUSES:
                count += 1
                continue
            if self._is_worker_alive_locked(record):
                count += 1
        return count

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


def _subagent_process_entry(
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
        subagent_id = str(payload.get("subagent_id", "")).strip()
        task = str(payload.get("task", "")).strip()
        context = str(payload.get("context", "")).strip()
        thread_id = str(payload.get("thread_id", "")).strip()
        subagent_name = str(payload.get("subagent_name", "")).strip()
        parent_context_raw = payload.get("parent_context")
        parent_context = (
            parent_context_raw if isinstance(parent_context_raw, dict) else None
        )
        prompt_registry = PromptRegistry(settings.prompts_dir)
        from llc.agent.graph import build_agent_graph

        agent = build_agent_graph(
            settings,
            role="subagent",
            prompt_registry=prompt_registry,
        )
        result = asyncio.run(
            _run_subagent_worker(
                agent=agent,
                task=task,
                context=context,
                thread_id=thread_id,
                subagent_id=subagent_id,
                subagent_name=subagent_name,
                settings=settings,
                prompt_registry=prompt_registry,
                parent_context=parent_context,
                stop_requested=lambda: _stop_event_is_set(stop_event),
                progress_callback=lambda progress: _emit_worker_event(
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
        result = {
            "status": "failed",
            "error": str(exc),
            "tool_calls": 0,
            "output_chars": 0,
            "usage_by_model": {},
        }
    _emit_worker_event(
        worker_queue,
        {
            "type": "result",
            "attempt": attempt,
            "result": result,
        },
    )


async def _run_subagent_worker(
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
                        content=_render_subagent_system_prompt(
                            settings,
                            prompt_registry,
                        )
                    ),
                    HumanMessage(
                        content=_render_subagent_task_message(
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
        try:
            try:
                while True:
                    if stop_requested():
                        raise _StopRequested("stop_requested")
                    elapsed = time.monotonic() - started_monotonic
                    if elapsed > max_runtime_s:
                        raise _StopRequested("max_runtime_exceeded")
                    next_chunk_timeout_s = _next_stream_timeout_s(
                        elapsed=elapsed,
                        max_runtime_s=max_runtime_s,
                        stall_timeout_s=stall_timeout_s,
                    )
                    if next_chunk_timeout_s <= 0:
                        raise _StopRequested("max_runtime_exceeded")
                    try:
                        mode, chunk = await asyncio.wait_for(
                            anext(stream),
                            timeout=next_chunk_timeout_s,
                        )
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        raise _StopRequested("stall_timeout_exceeded")

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
                                    raise _StopRequested("max_tool_calls_exceeded")
                                if tool_calls_batch:
                                    raw_tool_name = tool_calls_batch[-1].get("name", "")
                                    if (
                                        isinstance(raw_tool_name, str)
                                        and raw_tool_name.strip()
                                    ):
                                        last_tool_name = raw_tool_name.strip()
                                        current_activity = _activity_from_tool_name(
                                            last_tool_name
                                        )
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
                aclose = getattr(stream, "aclose", None)
                if callable(aclose):
                    try:
                        await aclose()
                    except Exception:
                        pass
        except _StopRequested as exc:
            status = "stuck" if _stop_reason_is_stuck(exc.reason) else "terminated"
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
            update_observation(
                trace_scope.observation,
                output={"status": "failed", "error": str(exc)},
            )
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
                "final_output_preview": _preview_text(
                    final_output,
                    _MAX_FINAL_OUTPUT_PREVIEW_CHARS,
                ),
            },
        )
        return result


def _emit_worker_event(worker_queue: Any, event: dict[str, Any]) -> None:
    try:
        worker_queue.put_nowait(event)
    except Exception:
        pass


def _stop_event_is_set(stop_event: Any) -> bool:
    if stop_event is None:
        return False
    try:
        return bool(stop_event.is_set())
    except Exception:
        return False


def _render_subagent_system_prompt(
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


def _render_subagent_task_message(
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


def _next_stream_timeout_s(
    *,
    elapsed: float,
    max_runtime_s: float,
    stall_timeout_s: float,
) -> float:
    remaining_runtime = max(max_runtime_s - elapsed, 0.0)
    if stall_timeout_s <= 0:
        return remaining_runtime
    return min(remaining_runtime, stall_timeout_s)


def _stop_reason_is_stuck(reason: str) -> bool:
    normalized = reason.strip()
    if not normalized:
        return False
    if normalized in _STUCK_STOP_REASONS:
        return True
    for stuck_reason in _STUCK_STOP_REASONS:
        if normalized.startswith(f"{stuck_reason}:"):
            return True
    return False


def _stuck_latest_report(reason: str) -> str:
    normalized = reason.strip()
    if normalized == "max_tool_calls_exceeded":
        return "Tool-call budget exceeded. Stopping."
    if normalized == "stall_timeout_exceeded":
        return "No heartbeat detected within stall timeout. Stopping."
    if normalized == "max_runtime_exceeded":
        return "Runtime limit exceeded. Stopping."
    if normalized.endswith(":unresponsive_after_stop"):
        return "Stop requested but worker stayed unresponsive. Marked stuck."
    return "Worker marked stuck."


def _stuck_activity_detail(reason: str) -> str:
    normalized = reason.strip()
    if normalized == "max_tool_calls_exceeded":
        return "Stopped after too many tool calls."
    if normalized == "stall_timeout_exceeded":
        return "Stopped after no progress heartbeat."
    if normalized == "max_runtime_exceeded":
        return "Stopped after hitting runtime limit."
    if normalized.endswith(":unresponsive_after_stop"):
        return "No response after stop request."
    return "Stopped due to watchdog safety limits."


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
