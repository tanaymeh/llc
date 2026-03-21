from __future__ import annotations

from concurrent.futures import Future
from threading import Event
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

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


class SubAgentRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    name: str
    base_task: str
    task: str
    context: str
    source: str
    thread_id: str
    parent_trace_id: str = ""
    parent_observation_id: str = ""
    parent_session_id: str = ""
    parent_turn_id: str = ""
    created_at: float
    updated_at: float
    status: SubAgentStatus = "running"
    attempt: int = 1
    stop_reason: str = ""
    final_output: str = ""
    completion_report: str = ""
    error: str = ""
    latest_report: str = ""
    current_activity: str = "Queued"
    activity_detail: str = ""
    last_tool_name: str = ""
    tool_calls: int = 0
    output_chars: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0
    last_activity_at: float = 0.0
    terminate_requested: bool = False
    pending_feedback: list[str] = Field(default_factory=list)
    stop_event: Event = Field(default_factory=Event)
    future: Future[dict[str, Any]] | None = None

