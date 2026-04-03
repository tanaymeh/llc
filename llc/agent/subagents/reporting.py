from __future__ import annotations

from typing import Any

from llc.agent.subagents.types import ACTIVE_STATUSES, SubAgentRecord

STUCK_STOP_REASONS = {
    "max_runtime_exceeded",
    "stall_timeout_exceeded",
    "max_tool_calls_exceeded",
}


def preview_text(text: str, limit: int) -> str:
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def activity_from_tool_name(tool_name: str) -> str:
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


def stop_reason_is_stuck(reason: str) -> bool:
    normalized = reason.strip()
    if not normalized:
        return False
    if normalized in STUCK_STOP_REASONS:
        return True
    for stuck_reason in STUCK_STOP_REASONS:
        if normalized.startswith(f"{stuck_reason}:"):
            return True
    return False


def stuck_latest_report(reason: str) -> str:
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


def stuck_activity_detail(reason: str) -> str:
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


def record_snapshot(
    record: SubAgentRecord,
    *,
    task_preview_chars: int,
    activity_preview_chars: int,
    report_preview_chars: int,
    error_preview_chars: int,
    final_output_preview_chars: int,
) -> dict[str, Any]:
    goal = preview_text(record.base_task, task_preview_chars)
    current_task = preview_text(record.task, task_preview_chars)
    if record.status in ACTIVE_STATUSES:
        current_activity = record.current_activity or "Running"
    elif record.status == "completed":
        current_activity = "Completed"
    elif record.status == "failed":
        failure = preview_text(record.error or "unknown error", activity_preview_chars)
        current_activity = f"Failed: {failure}" if failure else "Failed"
    elif record.status == "stuck":
        reason = preview_text(record.stop_reason or "watchdog stop", activity_preview_chars)
        current_activity = f"Stuck: {reason}" if reason else "Stuck"
    elif record.status == "terminated":
        reason = preview_text(record.stop_reason or "terminated", activity_preview_chars)
        current_activity = f"Terminated: {reason}" if reason else "Terminated"
    else:
        current_activity = "Inactive"
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
        "current_activity": preview_text(
            current_activity,
            activity_preview_chars,
        ),
        "activity_detail": preview_text(
            activity_detail,
            report_preview_chars,
        ),
        "last_tool_name": preview_text(
            record.last_tool_name,
            activity_preview_chars,
        ),
        "tool_calls": max(int(record.tool_calls or 0), 0),
        "total_tool_calls": max(int(record.total_tool_calls or 0), 0),
        "output_chars": max(int(record.output_chars or 0), 0),
        "created_at": float(record.created_at or 0),
        "updated_at": float(record.updated_at or 0),
        "started_at": float(record.started_at or 0),
        "finished_at": float(record.finished_at or 0),
    }

    latest_report = preview_text(record.latest_report, report_preview_chars)
    if latest_report:
        snapshot["latest_report"] = latest_report

    if record.status in ACTIVE_STATUSES:
        if latest_report:
            snapshot["progress"] = latest_report
        return snapshot

    if record.status == "completed":
        final_result = record.final_output.strip() or "Completed."
        snapshot["final_result"] = preview_text(
            final_result,
            final_output_preview_chars,
        )
        return snapshot

    if record.status == "failed":
        snapshot["error"] = preview_text(
            record.error or "unknown error",
            error_preview_chars,
        )
        return snapshot

    if record.stop_reason:
        snapshot["stop_reason"] = preview_text(
            record.stop_reason,
            error_preview_chars,
        )
    if latest_report:
        snapshot["progress"] = latest_report
    return snapshot
