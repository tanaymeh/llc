from __future__ import annotations

from typing import Any, Literal

from llc.models import AvailableModel
from llc.service.events import (
    CommandOutput,
    ErrorOccurred,
    Event,
    ReasoningDelta,
    SessionRestored,
    SubagentStatusUpdate,
    TextDelta,
    ToolCallStarted,
    ToolResultEvent,
    TurnCompleted,
    TurnStarted,
    UsageUpdate,
)
from llc.ui.state import (
    HealthStatus,
    LogCategory,
    LogEntry,
    MessageLaneEntry,
    MissionControlState,
    SessionPhase,
    ToolOutputEntry,
    ToolRunEntry,
    WorkerRow,
    build_model_options,
)

_CODE_TOOLS = {"Read", "Write", "Edit", "MultiEdit", "Grep", "Glob", "code_grep"}


def _normalize_reasoning_text(text: str) -> str:
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())


def _append_with_overlap(base: str, incoming: str) -> str:
    if not base:
        return incoming
    if incoming.startswith(base):
        return incoming
    if base.startswith(incoming):
        return base
    max_overlap = min(len(base), len(incoming))
    for overlap in range(max_overlap, 0, -1):
        if base.endswith(incoming[:overlap]):
            return f"{base} {incoming[overlap:]}".strip()
    return f"{base} {incoming}".strip()


def _format_tool_args(args: dict[str, Any]) -> str:
    if not args:
        return ""
    parts: list[str] = []
    for key, value in args.items():
        val = repr(value) if isinstance(value, str) else str(value)
        if len(val) > 56:
            val = val[:53] + "..."
        parts.append(f"{key}={val}")
    return ", ".join(parts)


class TelemetryMapper:
    def __init__(self, state: MissionControlState) -> None:
        self.state = state
        self._msg_index = 0
        self._reasoning_stream_text = ""

    def set_available_models(self, models: list[AvailableModel]) -> None:
        options = build_model_options(models)
        self.state.available_models = options
        self.state.model_selector.options = options
        self.state.model_selector.highlighted_index = 0

    def set_model_name(self, model_name: str) -> None:
        self.state.top_bar.model_name = model_name

    def set_sub_agent_mode(self, enabled: bool) -> None:
        self.state.top_bar.sub_agent_mode_enabled = enabled

    def set_busy(self, busy: bool) -> None:
        self.state.busy = busy
        if not busy and self.state.top_bar.phase == SessionPhase.VERIFY:
            self.state.top_bar.phase = SessionPhase.IDLE

    def add_user_message(self, text: str) -> None:
        entry = self._new_entry(
            role="user",
            content=text,
        )
        self.state.messages.append(entry)
        self.state.append_log(LogEntry(category=LogCategory.USER, message=text))

    def add_system_message(self, text: str, *, category: LogCategory = LogCategory.SYS) -> None:
        entry = self._new_entry(
            role="system",
            content=text,
        )
        self.state.messages.append(entry)
        self.state.append_log(LogEntry(category=category, message=text))

    def _new_entry(
        self,
        *,
        role: Literal["user", "agent", "system"],
        content: str,
        model_name: str | None = None,
    ) -> MessageLaneEntry:
        self._msg_index += 1
        return MessageLaneEntry(
            id=f"lane-{self._msg_index}",
            role=role,
            content=content,
            model_name=model_name,
        )

    def _ensure_active_agent_entry(self) -> MessageLaneEntry:
        active_id = self.state.active_message_id
        if active_id is not None:
            for entry in self.state.messages:
                if entry.id == active_id:
                    return entry
        entry = self._new_entry(
            role="agent",
            content="",
            model_name=self.state.top_bar.model_name,
        )
        self.state.messages.append(entry)
        self.state.active_message_id = entry.id
        return entry

    def process(self, event: Event) -> None:
        if isinstance(event, TurnStarted):
            self.state.top_bar.phase = SessionPhase.STREAMING
            self.state.top_bar.health = HealthStatus.STABLE
            self.state.clear_turn_buffers()
            self._reasoning_stream_text = ""
            self.state.active_message_id = None
            self.state.append_log(
                LogEntry(category=LogCategory.TURN, message=f"turn started: {event.turn_id}")
            )
            return

        if isinstance(event, TextDelta):
            agent = self._ensure_active_agent_entry()
            agent.content += event.text
            if agent.content.strip():
                agent.agent_status = ""
            self.state.top_bar.phase = SessionPhase.STREAMING
            return

        if isinstance(event, ReasoningDelta):
            agent = self._ensure_active_agent_entry()
            incoming = _normalize_reasoning_text(event.text)
            if not incoming:
                return
            self._reasoning_stream_text = _append_with_overlap(
                self._reasoning_stream_text,
                incoming,
            )
            self.state.set_reasoning_text(self._reasoning_stream_text, words_per_line=20)
            agent.reasoning_line = self._reasoning_stream_text
            if not agent.content.strip():
                agent.agent_status = "reasoning"
            return

        if isinstance(event, ToolCallStarted):
            args_preview = _format_tool_args(event.args)
            tool_entry = ToolRunEntry(
                tool_call_id=event.tool_call_id,
                tool_index=event.tool_index,
                tool_name=event.tool_name,
                args_preview=args_preview,
            )
            self.state.append_tool(tool_entry)
            agent = self._ensure_active_agent_entry()
            agent.tool_status = tool_entry.timeline_line
            self.state.top_bar.phase = SessionPhase.TOOLS
            category = (
                LogCategory.CODE if event.tool_name in _CODE_TOOLS else LogCategory.TOOL
            )
            self.state.append_log(
                LogEntry(
                    category=category,
                    source=event.tool_name,
                    message=tool_entry.timeline_line,
                )
            )
            return

        if isinstance(event, ToolResultEvent):
            if not event.user_facing:
                return
            agent = self._ensure_active_agent_entry()
            agent.tool_outputs.append(
                ToolOutputEntry(
                    tool_name=event.tool_name,
                    render_mode=event.render_mode,
                    content=event.content,
                )
            )
            self.state.append_log(
                LogEntry(
                    category=LogCategory.OBS,
                    source=event.tool_name,
                    message="user-facing tool output updated",
                )
            )
            return

        if isinstance(event, SubagentStatusUpdate):
            workers: list[WorkerRow] = []
            for worker in event.workers:
                if not isinstance(worker, dict):
                    continue
                workers.append(
                    WorkerRow(
                        id=str(worker.get("id", "")),
                        status=str(worker.get("status", "unknown")),
                        goal=str(worker.get("goal", "")),
                        current_task=str(worker.get("current_task", "")),
                        current_activity=str(worker.get("current_activity", "")),
                        activity_detail=str(worker.get("activity_detail", "")),
                        last_tool_name=str(worker.get("last_tool_name", "")),
                        tool_calls=int(worker.get("tool_calls", 0) or 0),
                        output_chars=int(worker.get("output_chars", 0) or 0),
                        updated_at=float(worker.get("updated_at", 0.0) or 0.0),
                    )
                )
            self.state.workers = workers
            self.state.top_bar.active_subagents = max(event.active_count, 0)
            self.state.top_bar.total_workers = len(workers)
            return

        if isinstance(event, UsageUpdate):
            self.state.top_bar.session_input_tokens = max(event.session_input_tokens, 0)
            self.state.top_bar.session_output_tokens = max(event.session_output_tokens, 0)
            self.state.top_bar.session_cost = max(event.session_cost, 0.0)
            return

        if isinstance(event, CommandOutput):
            if event.message:
                entry = self._new_entry(
                    role="agent",
                    content=event.message,
                    model_name=self.state.top_bar.model_name,
                )
                self.state.messages.append(entry)
                self.state.append_log(LogEntry(category=LogCategory.AGENT, message=event.message))
            return

        if isinstance(event, ErrorOccurred):
            self.state.top_bar.health = HealthStatus.ERROR
            self.state.top_bar.phase = SessionPhase.IDLE
            self.add_system_message(f"Request failed: {event.message}", category=LogCategory.ERR)
            self.state.active_message_id = None
            return

        if isinstance(event, SessionRestored):
            self.state.append_log(
                LogEntry(
                    category=LogCategory.SYS,
                    message=f"session restored: {event.session_id} ({event.message_count} msgs)",
                )
            )
            return

        if isinstance(event, TurnCompleted):
            if self.state.active_message_id is not None:
                active = self._ensure_active_agent_entry()
                if active.reasoning_line and not active.reasoning_summary:
                    active.reasoning_summary = active.reasoning_line
                active.reasoning_line = ""
                active.tool_status = ""
                active.agent_status = ""
            self.state.active_message_id = None
            self.state.top_bar.phase = SessionPhase.VERIFY
            if event.compact_message:
                self.state.append_log(
                    LogEntry(
                        category=LogCategory.SYS,
                        message=event.compact_message,
                    )
                )
            return
