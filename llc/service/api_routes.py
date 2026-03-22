from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from llc.service.api_engine_manager import EngineManager
from llc.service.api_models import (
    CreateSessionRequest,
    RestoreSessionResponse,
    SessionRecordResponse,
    SessionSummaryResponse,
    session_summary,
)


def register_api_routes(app: FastAPI, manager: EngineManager) -> None:
    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"ok": True}

    @app.get("/api/sessions", response_model=list[SessionRecordResponse])
    async def list_sessions() -> list[SessionRecordResponse]:
        return await manager.list_sessions()

    @app.post("/api/sessions", response_model=SessionSummaryResponse)
    async def create_session(
        payload: CreateSessionRequest | None = None,
    ) -> SessionSummaryResponse:
        engine = await manager.get_or_create(payload.session_id if payload else None)
        return session_summary(engine)

    @app.get("/api/sessions/{session_id}", response_model=SessionSummaryResponse)
    async def get_session(session_id: str) -> SessionSummaryResponse:
        engine = await manager.get_or_create(session_id)
        return session_summary(engine)

    @app.post("/api/sessions/{session_id}/restore", response_model=RestoreSessionResponse)
    async def restore_session(session_id: str) -> RestoreSessionResponse:
        engine = await manager.get_or_create(session_id)
        events: list[dict[str, Any]] = []
        async for event in engine.restore_session(session_id):
            events.append(event.model_dump(mode="json"))
        return RestoreSessionResponse(session=session_summary(engine), events=events)

    @app.get("/api/sessions/{session_id}/subagents")
    async def get_subagent_report(session_id: str) -> dict[str, Any]:
        engine = await manager.get_or_create(session_id)
        return engine.get_subagent_report(include_all=True)
