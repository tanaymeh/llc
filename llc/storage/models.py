from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

MessageRole = Literal["user", "assistant", "system", "tool"]


class SessionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    title: str = ""
    model_name: str
    sub_agent_mode: bool = False
    created_at: float
    updated_at: float


class MessageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    session_id: str
    role: MessageRole
    content: str
    tool_calls_json: str = "[]"
    usage_input_tokens: int = 0
    usage_output_tokens: int = 0
    created_at: float


class TokenUsageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: int | None = None
    session_id: str
    model_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    recorded_at: float


class EventRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: int | None = None
    session_id: str
    turn_id: str = ""
    event_type: str
    payload_json: str
    created_at: float

