from __future__ import annotations

import logging
import multiprocessing as mp
import sys
import time
import uuid
from queue import Empty
from threading import RLock
from typing import Any, Callable

from llc.agent.subagents.coordination import AgentCoordinationLayer
from llc.agent.subagents.reporting import (
    record_snapshot,
    preview_text,
    stop_reason_is_stuck,
    stuck_activity_detail,
    stuck_latest_report,
)
from llc.agent.subagents.types import ACTIVE_STATUSES, SubAgentRecord
from llc.agent.subagents.usage import normalize_usage_by_model
from llc.agent.subagents.worker_runner import (
    subagent_process_entry,
    task_with_feedback,
)
from llc.config import Settings
from llc.logging_utils import render_kv
from llc.observability import capture_parent_context
from llc.service.prompt_registry import PromptRegistry

SubAgentBuilder = Callable[[Settings], Any]
_MAX_TASK_PREVIEW_CHARS = 180
_MAX_REPORT_PREVIEW_CHARS = 220
_MAX_ERROR_PREVIEW_CHARS = 220
_MAX_FINAL_OUTPUT_PREVIEW_CHARS = 600
_MAX_REPORT_WORKERS = 8
_MAX_ACTIVITY_PREVIEW_CHARS = 80
_WAIT_POLL_INTERVAL_S = 0.1
_TOOL_LOOP_LOG_INTERVAL = 5
_COORDINATION_TOOL_NAMES = {
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
_LOGGER = logging.getLogger("llc.subagents.runtime")


def _coordination_config(settings: Settings) -> dict[str, Any]:
    return {
        "debug_logging": bool(settings.sub_agent_debug_logging),
        "team_status_interval_cycles": int(
            settings.sub_agent_team_status_interval_cycles
        ),
        "shared_notes_interval_cycles": int(
            settings.sub_agent_shared_notes_interval_cycles
        ),
        "lock_default_lease_s": int(settings.sub_agent_lock_default_lease_s),
        "lock_renew_s": int(settings.sub_agent_lock_renew_s),
        "lock_near_expiry_s": int(settings.sub_agent_lock_near_expiry_s),
        "shared_notes_max_entries": int(settings.sub_agent_shared_notes_max_entries),
        "inbox_read_max": int(settings.sub_agent_inbox_read_max),
    }


class SubAgentRuntime:
    def __init__(
        self,
        settings: Settings,
        build_subagent: SubAgentBuilder,
        prompt_registry: PromptRegistry | None = None,
        launch_precheck: Callable[[], str | None] | None = None,
    ) -> None:
        self._settings = settings
        self._build_subagent = build_subagent
        self._prompt_registry = prompt_registry
        self._launch_precheck = launch_precheck
        self._records: dict[str, SubAgentRecord] = {}
        self._lock = RLock()
        self._closed = False
        self._usage_input_tokens = 0
        self._usage_output_tokens = 0
        self._usage_by_model: dict[str, dict[str, int]] = {}
        self._last_logged_tool_calls: dict[str, int] = {}
        self._last_logged_tool_name: dict[str, str] = {}
        self._same_tool_streak: dict[str, int] = {}
        main_file = getattr(sys.modules.get("__main__"), "__file__", "") or ""
        prefer_spawn = bool(main_file and not str(main_file).startswith("<"))
        if prefer_spawn:
            self._mp_context = mp.get_context("spawn")
        else:
            try:
                self._mp_context = mp.get_context("fork")
            except ValueError:
                self._mp_context = mp.get_context("spawn")
        try:
            self._coordination: AgentCoordinationLayer | None = AgentCoordinationLayer.create(
                self._mp_context,
                config=_coordination_config(settings),
            )
        except Exception:
            self._coordination = None

    def _debug_enabled(self) -> bool:
        return bool(self._settings.sub_agent_debug_logging)

    def _debug_log(self, event: str, **fields: Any) -> None:
        if not self._debug_enabled():
            return
        kv = render_kv(fields)
        if kv:
            _LOGGER.info("[subagent-debug] event=%s %s", event, kv)
            return
        _LOGGER.info("[subagent-debug] event=%s", event)

    def update_settings(self, settings: Settings) -> None:
        with self._lock:
            self._settings = settings
            if self._coordination is not None:
                self._coordination.update_config(_coordination_config(settings))
            self._debug_log(
                "runtime_settings_updated",
                sub_agent_mode_enabled=settings.sub_agent_mode_enabled,
                max_sub_agents=settings.max_sub_agents,
                debug_logging=settings.sub_agent_debug_logging,
            )

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._debug_log("runtime_shutdown_start")
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
                    if self._coordination is not None:
                        self._coordination.mark_agent_stopped(
                            agent_id=record.id,
                            status="terminated",
                        )
            if self._coordination is not None:
                self._coordination.shutdown()
                self._coordination = None
            self._debug_log("runtime_shutdown_complete")

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
            if self._launch_precheck is not None:
                try:
                    launch_error = (self._launch_precheck() or "").strip()
                except Exception:
                    launch_error = "Sub-agent launch preflight failed."
                if launch_error:
                    self._debug_log(
                        "launch_subagent_blocked",
                        source=source,
                        reason=launch_error,
                    )
                    return {"ok": False, "error": launch_error}
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
            if not provided_name:
                return {
                    "ok": False,
                    "error": "Sub-agent name is required for launch.",
                }
            selected_name = provided_name
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
            if self._coordination is not None:
                self._coordination.register_agent(
                    agent_id=subagent_id,
                    name=selected_name,
                    task=clean_task,
                    spawned_by=source,
                    status="running",
                )
            self._submit_locked(record)
            self._debug_log(
                "launch_subagent",
                subagent_id=subagent_id,
                name=selected_name,
                source=source,
                active_count=self._active_count_locked(),
                max_sub_agents=self._settings.max_sub_agents,
            )
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
                record.activity_detail = preview_text(
                    clean_feedback,
                    _MAX_REPORT_PREVIEW_CHARS,
                )
                record.updated_at = time.time()
                record.stop_requested_at = record.updated_at
                self._request_worker_stop_locked(record)
                if self._coordination is not None:
                    self._coordination.update_agent(
                        agent_id=subagent_id,
                        status="restarting",
                        task=record.task,
                        heartbeat=True,
                    )
            else:
                self._restart_with_feedback_locked(record)

            snapshot = self._record_snapshot_locked(record)
            snapshot["ok"] = True
            self._debug_log(
                "revise_subagent",
                subagent_id=subagent_id,
                source=source,
                status=record.status,
            )
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
                    record.activity_detail = preview_text(
                        reason.strip(),
                        _MAX_REPORT_PREVIEW_CHARS,
                    )
                record.updated_at = time.time()
                record.stop_requested_at = record.updated_at
                self._request_worker_stop_locked(record)
                if self._coordination is not None:
                    self._coordination.update_agent(
                        agent_id=subagent_id,
                        status=record.status,
                        heartbeat=True,
                    )
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
                if self._coordination is not None:
                    self._coordination.mark_agent_stopped(
                        agent_id=subagent_id,
                        status="terminated",
                    )

            snapshot = self._record_snapshot_locked(record)
            snapshot["ok"] = True
            self._debug_log(
                "terminate_subagent",
                subagent_id=subagent_id,
                status=record.status,
                reason=reason.strip(),
            )
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
            "coordination_handles": (
                self._coordination.export_handles()
                if self._coordination is not None
                else None
            ),
        }
        process = self._mp_context.Process(
            target=subagent_process_entry,
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
            record.activity_detail = preview_text(
                record.error or "unknown error",
                _MAX_REPORT_PREVIEW_CHARS,
            )
            record.current_activity = "Agent de-spawned"
            record.finished_at = time.time()
            self._close_worker_queue_safely(worker_queue)
            if self._coordination is not None:
                self._coordination.mark_agent_stopped(
                    agent_id=record.id,
                    status="failed",
                )
            self._debug_log(
                "worker_start_failed",
                subagent_id=record.id,
                error=str(exc),
            )
            return
        record.worker_process = process
        record.worker_queue = worker_queue
        self._last_logged_tool_calls[record.id] = 0
        self._last_logged_tool_name[record.id] = ""
        self._same_tool_streak[record.id] = 0
        self._debug_log(
            "worker_started",
            subagent_id=record.id,
            attempt=record.attempt,
            thread_id=record.thread_id,
        )
        if self._coordination is not None:
            self._coordination.update_agent(
                agent_id=record.id,
                status="running",
                task=record.task,
                heartbeat=True,
            )

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
            total_tool_calls = max(
                int(event.get("total_tool_calls", tool_calls) or 0),
                0,
            )
            output_chars = max(int(event.get("output_chars", 0) or 0), 0)
            report = str(event.get("report", "") or "")
            activity = str(event.get("activity", "") or "")
            activity_detail = str(event.get("activity_detail", "") or "")
            last_tool_name = str(event.get("last_tool_name", "") or "")
            last_tool_summary = str(event.get("last_tool_summary", "") or "")
            record.updated_at = now
            record.last_activity_at = now
            record.tool_calls = max(record.tool_calls, tool_calls)
            record.total_tool_calls = max(record.total_tool_calls, total_tool_calls)
            record.output_chars = max(record.output_chars, output_chars)
            if report:
                record.latest_report = preview_text(report, _MAX_REPORT_PREVIEW_CHARS)
            if activity:
                record.current_activity = preview_text(activity, _MAX_ACTIVITY_PREVIEW_CHARS)
            if activity_detail:
                record.activity_detail = preview_text(
                    activity_detail,
                    _MAX_REPORT_PREVIEW_CHARS,
                )
            if last_tool_name:
                record.last_tool_name = preview_text(
                    last_tool_name,
                    _MAX_ACTIVITY_PREVIEW_CHARS,
                )
            previous_tool_calls = int(self._last_logged_tool_calls.get(record.id, 0) or 0)
            if tool_calls > previous_tool_calls:
                self._last_logged_tool_calls[record.id] = tool_calls
                current_tool_name = (last_tool_name or "").strip()
                previous_tool_name = self._last_logged_tool_name.get(record.id, "")
                if current_tool_name and current_tool_name == previous_tool_name:
                    streak = int(self._same_tool_streak.get(record.id, 1) or 1) + 1
                else:
                    streak = 1
                self._same_tool_streak[record.id] = streak
                self._last_logged_tool_name[record.id] = current_tool_name
                is_coordination_tool = current_tool_name in _COORDINATION_TOOL_NAMES
                is_first_tool_log = previous_tool_name == ""
                is_non_coordination_loop_tick = (
                    not is_coordination_tool
                    and current_tool_name == previous_tool_name
                    and streak % _TOOL_LOOP_LOG_INTERVAL == 0
                )
                should_log = (
                    is_coordination_tool
                    or is_first_tool_log
                    or is_non_coordination_loop_tick
                )
                if should_log:
                    self._debug_log(
                        "worker_tool_call",
                        subagent_id=record.id,
                        attempt=record.attempt,
                        tool_name=current_tool_name or "unknown",
                        tool_summary=last_tool_summary or None,
                        same_tool_streak=streak if streak > 1 else None,
                        tool_calls=tool_calls,
                        total_tool_calls=total_tool_calls,
                        activity=record.current_activity,
                    )
            if self._coordination is not None:
                self._coordination.update_agent(
                    agent_id=record.id,
                    status=record.status,
                    task=record.task,
                    heartbeat=True,
                )
            return
        if event_type == "result":
            payload = event.get("result")
            result = payload if isinstance(payload, dict) else {}
            record.worker_result_received = True
            self._debug_log(
                "worker_result_received",
                subagent_id=record.id,
                attempt=record.attempt,
                status=str(result.get("status", "")).strip(),
            )
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
        record.total_tool_calls = max(
            record.total_tool_calls,
            int(
                result.get(
                    "total_tool_calls",
                    result.get("tool_calls", 0),
                )
                or 0
            ),
        )
        record.output_chars = max(
            record.output_chars,
            int(result.get("output_chars", 0) or 0),
        )
        self._merge_usage_locked(normalize_usage_by_model(result.get("usage_by_model")))
        self._cleanup_worker_handles_locked(record)

        if record.pending_feedback and not record.terminate_requested:
            self._restart_with_feedback_locked(record)
            return

        if record.status == "completed":
            record.latest_report = "Completed."
            record.activity_detail = "Completed successfully."
        elif record.status == "failed":
            record.latest_report = f"Failed: {record.error or 'unknown error'}"
            record.activity_detail = preview_text(
                record.error or "unknown error",
                _MAX_REPORT_PREVIEW_CHARS,
            )
        elif record.status == "stuck":
            record.latest_report = stuck_latest_report(record.stop_reason)
            record.activity_detail = stuck_activity_detail(record.stop_reason)
        else:
            record.latest_report = "Stopped."
            record.activity_detail = "Stopped before completion."
        record.current_activity = "Agent de-spawned"
        self._last_logged_tool_calls.pop(record.id, None)
        self._last_logged_tool_name.pop(record.id, None)
        self._same_tool_streak.pop(record.id, None)
        self._debug_log(
            "worker_finished",
            subagent_id=record.id,
            attempt=record.attempt,
            status=record.status,
            stop_reason=record.stop_reason,
            error=record.error,
            tool_calls=record.tool_calls,
            total_tool_calls=record.total_tool_calls,
        )
        if self._coordination is not None:
            terminal_status = record.status if record.status in {"completed", "failed"} else "terminated"
            self._coordination.mark_agent_stopped(
                agent_id=record.id,
                status=terminal_status,
            )

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
        record.task = task_with_feedback(
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
            record.activity_detail = preview_text(feedback, _MAX_REPORT_PREVIEW_CHARS)
        record.last_tool_name = ""
        self._debug_log(
            "worker_restarting",
            subagent_id=record.id,
            attempt=record.attempt,
        )
        if self._coordination is not None:
            self._coordination.update_agent(
                agent_id=record.id,
                status="restarting",
                task=record.task,
                heartbeat=True,
            )
        self._submit_locked(record)

    def _refresh_stuck_locked(self) -> None:
        now = time.time()
        if self._coordination is not None:
            self._coordination.sweep_expired_locks()
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
                        "stuck" if stop_reason_is_stuck(fallback_stop_reason) else "terminated"
                    )
                    fallback = {
                        "status": fallback_status,
                        "stop_reason": fallback_stop_reason,
                        "tool_calls": record.tool_calls,
                        "total_tool_calls": record.total_tool_calls,
                        "output_chars": record.output_chars,
                        "usage_by_model": {},
                    }
                else:
                    fallback = {
                        "status": "failed",
                        "error": "Worker process exited without a result payload.",
                        "tool_calls": record.tool_calls,
                        "total_tool_calls": record.total_tool_calls,
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
                record.latest_report = stuck_latest_report(stop_reason)
                record.current_activity = "Terminating"
                record.activity_detail = stuck_activity_detail(stop_reason)
                record.updated_at = now
                record.stop_requested_at = now
                self._request_worker_stop_locked(record)
                self._debug_log(
                    "watchdog_stop_requested",
                    subagent_id=record.id,
                    reason=stop_reason,
                    elapsed_s=round(elapsed, 2),
                    idle_for_s=round(idle_for, 2),
                )
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
                    "total_tool_calls": record.total_tool_calls,
                    "output_chars": record.output_chars,
                    "usage_by_model": {},
                }
            elif base_reason.startswith("terminated"):
                forced = {
                    "status": "terminated",
                    "stop_reason": base_reason,
                    "tool_calls": record.tool_calls,
                    "total_tool_calls": record.total_tool_calls,
                    "output_chars": record.output_chars,
                    "usage_by_model": {},
                }
            else:
                forced = {
                    "status": "stuck",
                    "stop_reason": f"{base_reason}:unresponsive_after_stop",
                    "tool_calls": record.tool_calls,
                    "total_tool_calls": record.total_tool_calls,
                    "output_chars": record.output_chars,
                    "usage_by_model": {},
                }
            self._apply_worker_result_locked(record, forced)
            self._debug_log(
                "watchdog_forced_termination",
                subagent_id=record.id,
                reason=forced.get("stop_reason"),
            )

    def _record_snapshot_locked(self, record: SubAgentRecord) -> dict[str, Any]:
        snapshot = record_snapshot(
            record,
            task_preview_chars=_MAX_TASK_PREVIEW_CHARS,
            activity_preview_chars=_MAX_ACTIVITY_PREVIEW_CHARS,
            report_preview_chars=_MAX_REPORT_PREVIEW_CHARS,
            error_preview_chars=_MAX_ERROR_PREVIEW_CHARS,
            final_output_preview_chars=_MAX_FINAL_OUTPUT_PREVIEW_CHARS,
        )
        if self._coordination is not None:
            snapshot.update(self._coordination.coordination_snapshot_for_agent(record.id))
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
        return "worker"
    if len(slug) <= 24:
        return slug
    return slug[:24].rstrip("-")


def _build_subagent_id(name: str) -> str:
    slug = _slugify_label(name)
    return f"subagent-{slug}-{uuid.uuid4().hex[:4]}"
