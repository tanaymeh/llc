from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from llc.models import AvailableModel
from llc.service.engine import SessionEngine

DEFAULT_WS_PATH_TEMPLATE = "/api/ws/{session_id}"


class CreateSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str | None = None


class SessionSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    model_name: str
    sub_agent_mode_enabled: bool
    max_sub_agents: int
    available_models: list[AvailableModel] = Field(default_factory=list)
    ws_path: str


class RestoreSessionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session: SessionSummaryResponse
    events: list[dict[str, Any]] = Field(default_factory=list)


class SessionRecordResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    model_name: str
    sub_agent_mode: bool
    created_at: float
    updated_at: float


class WebSocketMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["send_message", "interrupt", "restore_session"]
    text: str = ""
    session_id: str | None = None


def session_summary(engine: SessionEngine) -> SessionSummaryResponse:
    return SessionSummaryResponse(
        session_id=engine.session_id,
        model_name=engine.model_name,
        sub_agent_mode_enabled=engine.sub_agent_mode_enabled,
        max_sub_agents=engine.get_subagent_report().get("max_sub_agents", 0),
        available_models=engine.available_models,
        ws_path=DEFAULT_WS_PATH_TEMPLATE.format(session_id=engine.session_id),
    )
