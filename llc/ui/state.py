from __future__ import annotations

import time
from collections import deque
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from llc.models import AvailableModel

MAX_LOG_ENTRIES = 500
MAX_TOOL_TIMELINE_ENTRIES = 64
MAX_REASONING_ENTRIES = 120


class UiMode(str, Enum):
    OVERVIEW = "overview"
    FOCUS_LOGS = "focus_logs"
    FOCUS_AGENTS = "focus_agents"
    FOCUS_EXECUTION = "focus_execution"


class SessionPhase(str, Enum):
    IDLE = "idle"
    STREAMING = "streaming"
    TOOLS = "tools"
    VERIFY = "verify"


class HealthStatus(str, Enum):
    STABLE = "stable"
    WARNING = "warning"
    ERROR = "error"


class LogCategory(str, Enum):
    SYS = "SYS"
    USER = "USER"
    AGENT = "AGENT"
    TURN = "TURN"
    TOOL = "TOOL"
    CODE = "CODE"
    OBS = "OBS"
    WARN = "WARN"
    ERR = "ERR"


class ModelOption(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    name: str
    context_length: int | None = None

    @computed_field
    @property
    def display_label(self) -> str:
        if self.context_length is None:
            return f"{self.id}"
        return f"{self.id} ({self.context_length:,} ctx)"


class ModelSelectorState(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    visible: bool = False
    query: str = ""
    highlighted_index: int = 0
    options: list[ModelOption] = Field(default_factory=list)

    @computed_field
    @property
    def filtered_options(self) -> list[ModelOption]:
        q = self.query.strip().lower()
        if not q:
            return self.options
        return [
            option
            for option in self.options
            if q in option.id.lower() or q in option.name.lower()
        ]

    @computed_field
    @property
    def has_options(self) -> bool:
        return bool(self.filtered_options)

    @field_validator("highlighted_index")
    @classmethod
    def _validate_index(cls, value: int) -> int:
        return max(value, 0)


class TopBarState(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    model_name: str = "unknown"
    session_input_tokens: int = 0
    session_output_tokens: int = 0
    session_cost: float = 0.0
    sub_agent_mode_enabled: bool = False
    active_subagents: int = 0
    total_workers: int = 0
    phase: SessionPhase = SessionPhase.IDLE
    health: HealthStatus = HealthStatus.STABLE
    started_at: float = Field(default_factory=time.time)

    @computed_field
    @property
    def elapsed_label(self) -> str:
        elapsed = max(int(time.time() - self.started_at), 0)
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"

    @computed_field
    @property
    def tokens_label(self) -> str:
        return f"{self.session_input_tokens:,}/{self.session_output_tokens:,}"


class ToolRunEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    tool_call_id: str
    tool_index: int
    tool_name: str
    args_preview: str = ""
    created_at: float = Field(default_factory=time.time)

    @computed_field
    @property
    def timeline_line(self) -> str:
        if not self.args_preview:
            return f"[{self.tool_index}] {self.tool_name}"
        return f"[{self.tool_index}] {self.tool_name}({self.args_preview})"


class ToolOutputEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    tool_name: str
    render_mode: str = ""
    content: str


class MessageLaneEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str
    role: Literal["user", "agent", "system"]
    content: str = ""
    model_name: str | None = None
    reasoning_line: str = ""
    reasoning_summary: str = ""
    tool_status: str = ""
    agent_status: str = ""
    tool_outputs: list[ToolOutputEntry] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)

    @computed_field
    @property
    def title(self) -> str:
        if self.role == "user":
            return "USER"
        if self.role == "system":
            return "SYS"
        return "AGENT"

    @computed_field
    @property
    def timestamp_label(self) -> str:
        return datetime.fromtimestamp(self.created_at).strftime("%H:%M:%S")


class WorkerRow(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str
    status: str = "unknown"
    goal: str = ""
    current_task: str = ""
    current_activity: str = ""
    activity_detail: str = ""
    last_tool_name: str = ""
    tool_calls: int = 0
    output_chars: int = 0
    updated_at: float = 0.0

    @field_validator("status", mode="before")
    @classmethod
    def _normalize_status(cls, value: object) -> str:
        return str(value or "unknown").strip().lower()

    @computed_field
    @property
    def short_id(self) -> str:
        return self.id.removeprefix("subagent-")

    @computed_field
    @property
    def is_active(self) -> bool:
        return self.status in {"running", "restarting", "terminating"}

    @computed_field
    @property
    def status_label(self) -> str:
        return self.status.upper()


class LogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    category: LogCategory
    message: str
    source: str = ""
    created_at: float = Field(default_factory=time.time)

    @field_validator("message")
    @classmethod
    def _normalize_message(cls, value: str) -> str:
        compact = " ".join(value.split())
        return compact if compact else "-"

    @computed_field
    @property
    def timestamp_label(self) -> str:
        return datetime.fromtimestamp(self.created_at).strftime("%H:%M:%S")

    @computed_field
    @property
    def line(self) -> str:
        if self.source:
            return f"{self.timestamp_label} {self.category.value:<4} {self.source:<8} {self.message}"
        return f"{self.timestamp_label} {self.category.value:<4} {self.message}"


class MissionControlState(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    top_bar: TopBarState = Field(default_factory=TopBarState)
    mode: UiMode = UiMode.OVERVIEW
    busy: bool = False
    messages: list[MessageLaneEntry] = Field(default_factory=list)
    active_message_id: str | None = None
    tool_timeline: list[ToolRunEntry] = Field(default_factory=list)
    reasoning_lines: deque[str] = Field(default_factory=deque)
    workers: list[WorkerRow] = Field(default_factory=list)
    logs: deque[LogEntry] = Field(default_factory=deque)
    model_selector: ModelSelectorState = Field(default_factory=ModelSelectorState)
    available_models: list[ModelOption] = Field(default_factory=list)

    def append_log(self, entry: LogEntry) -> None:
        self.logs.append(entry)
        while len(self.logs) > MAX_LOG_ENTRIES:
            self.logs.popleft()

    def append_tool(self, entry: ToolRunEntry) -> None:
        self.tool_timeline.append(entry)
        if len(self.tool_timeline) > MAX_TOOL_TIMELINE_ENTRIES:
            self.tool_timeline = self.tool_timeline[-MAX_TOOL_TIMELINE_ENTRIES:]

    def set_reasoning_text(self, text: str, *, words_per_line: int = 20) -> None:
        if words_per_line <= 0:
            words_per_line = 20
        normalized = " ".join(text.replace("\r", " ").replace("\n", " ").split())
        if not normalized:
            self.reasoning_lines.clear()
            return
        words = normalized.split(" ")
        wrapped = [
            " ".join(words[index : index + words_per_line])
            for index in range(0, len(words), words_per_line)
        ]
        self.reasoning_lines = deque(wrapped[-MAX_REASONING_ENTRIES:])

    def clear_turn_buffers(self) -> None:
        self.tool_timeline = []
        self.reasoning_lines.clear()


def build_model_options(models: list[AvailableModel]) -> list[ModelOption]:
    options: list[ModelOption] = []
    for model in models:
        options.append(
            ModelOption(
                id=model.id,
                name=model.name,
                context_length=model.context_length,
            )
        )
    return options
