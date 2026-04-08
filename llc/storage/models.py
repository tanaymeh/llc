from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

MessageRole = Literal["user", "assistant", "system", "tool"]
ActorKind = Literal[
    "orchestrator",
    "subagent",
    "hook",
    "maintenance",
    "system",
]
MessageKind = Literal[
    "chat",
    "tool_call",
    "tool_result",
    "command",
    "checkpoint",
    "history_snapshot",
]


class ConversationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    title: str = ""
    model_name: str
    sub_agent_mode: bool = False
    created_at: float
    updated_at: float
    legacy_session_id: str = ""


class ConversationMessageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    conversation_id: str
    actor_kind: ActorKind
    actor_id: str = ""
    turn_id: str = ""
    message_kind: MessageKind = "chat"
    role: MessageRole
    content_json: str
    tool_call_id: str = ""
    langfuse_trace_id: str = ""
    visible_to_orchestrator: bool = False
    created_at: float


class ConversationUsageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: int | None = None
    conversation_id: str
    actor_kind: ActorKind = "orchestrator"
    actor_id: str = ""
    model_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    recorded_at: float


class ConversationEventRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: int | None = None
    conversation_id: str
    sequence_no: int | None = None
    actor_kind: ActorKind = "orchestrator"
    actor_id: str = ""
    turn_id: str = ""
    event_type: str
    payload_json: str
    langfuse_trace_id: str = ""
    created_at: float


class ConversationStateSnapshotRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    conversation_id: str
    state_kind: str
    payload_json: str
    updated_at: float


# Compatibility aliases while the service/API layer still uses session terminology.
SessionRecord = ConversationRecord
MessageRecord = ConversationMessageRecord
TokenUsageRecord = ConversationUsageRecord
EventRecord = ConversationEventRecord
