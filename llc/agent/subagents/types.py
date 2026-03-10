from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field
from threading import Event
from typing import Any, Literal

SubAgentStatus = Literal[
    "running",
    "completed",
    "failed",
    "restarting",
    "terminating",
    "terminated",
    "stuck",
]

ACTIVE_STATUSES: set[SubAgentStatus] = {"running", "restarting", "terminating"}


@dataclass(slots=True)
class SubAgentRecord:
    id: str
    name: str
    base_task: str
    task: str
    context: str
    source: str
    thread_id: str
    created_at: float
    updated_at: float
    status: SubAgentStatus = "running"
    attempt: int = 1
    stop_reason: str = ""
    final_output: str = ""
    completion_report: str = ""
    error: str = ""
    latest_report: str = ""
    tool_calls: int = 0
    output_chars: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0
    last_activity_at: float = 0.0
    terminate_requested: bool = False
    pending_feedback: list[str] = field(default_factory=list)
    stop_event: Event = field(default_factory=Event)
    future: Future[dict[str, Any]] | None = None

