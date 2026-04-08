from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, TypeAdapter


class _EventBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: str


class TurnStarted(_EventBase):
    type: Literal["turn_started"] = "turn_started"
    turn_id: str


class TextDelta(_EventBase):
    type: Literal["text_delta"] = "text_delta"
    text: str


class ReasoningDelta(_EventBase):
    type: Literal["reasoning_delta"] = "reasoning_delta"
    text: str


class ToolCallStarted(_EventBase):
    type: Literal["tool_call_started"] = "tool_call_started"
    tool_call_id: str
    tool_name: str
    tool_index: int
    args: dict[str, Any] = Field(default_factory=dict)


class ToolResultEvent(_EventBase):
    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    tool_name: str
    content: str
    user_facing: bool = False
    render_mode: str = ""


class UsageUpdate(_EventBase):
    type: Literal["usage_update"] = "usage_update"
    turn_input_tokens: int = 0
    turn_output_tokens: int = 0
    session_input_tokens: int = 0
    session_output_tokens: int = 0
    session_cost: float = 0.0


class SubagentStatusUpdate(_EventBase):
    type: Literal["subagent_status"] = "subagent_status"
    workers: list[dict[str, Any]] = Field(default_factory=list)
    active_count: int = 0
    max_sub_agents: int = 0


HookStage = Literal["queued", "running", "applied", "failed", "coalesced", "skipped"]


class HookUpdate(_EventBase):
    type: Literal["hook_update"] = "hook_update"
    hook_name: str
    stage: HookStage
    turn_id: str | None = None
    message: str | None = None


class TurnCompleted(_EventBase):
    type: Literal["turn_completed"] = "turn_completed"
    input_tokens: int = 0
    output_tokens: int = 0
    compact_message: str | None = None


class CommandOutput(_EventBase):
    type: Literal["command_output"] = "command_output"
    message: str | None = None
    should_exit: bool = False
    data: dict[str, Any] = Field(default_factory=dict)


class ErrorOccurred(_EventBase):
    type: Literal["error"] = "error"
    message: str


class SessionRestored(_EventBase):
    type: Literal["session_restored"] = "session_restored"
    conversation_id: str = ""
    session_id: str
    message_count: int = 0


Event = Annotated[
    (
        TurnStarted
        | TextDelta
        | ReasoningDelta
        | ToolCallStarted
        | ToolResultEvent
        | UsageUpdate
        | SubagentStatusUpdate
        | HookUpdate
        | TurnCompleted
        | CommandOutput
        | ErrorOccurred
        | SessionRestored
    ),
    Discriminator("type"),
]

EVENT_ADAPTER = TypeAdapter(Event)
