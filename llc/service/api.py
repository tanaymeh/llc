from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager, suppress
from typing import Any, Literal

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llc.commands import CommandRegistry
from llc.config import Settings
from llc.models import AvailableModel
from llc.service.engine import SessionEngine
from llc.service.events import ErrorOccurred, Event, SubagentStatusUpdate
from llc.storage.store import SessionStore

_DEFAULT_WS_PATH_TEMPLATE = "/api/ws/{session_id}"


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


class EngineManager:
    def __init__(self, settings: Settings, registry: CommandRegistry) -> None:
        self._settings = settings
        self._registry = registry
        self._engines: dict[str, SessionEngine] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(self, session_id: str | None = None) -> SessionEngine:
        requested = (session_id or "").strip()
        async with self._lock:
            if requested and requested in self._engines:
                return self._engines[requested]

            engine = SessionEngine(
                self._settings,
                self._registry,
                session_id=requested or None,
                store=SessionStore(self._settings.db_path),
            )
            await engine.initialize()
            self._engines[engine.session_id] = engine
            return engine

    async def list_sessions(self) -> list[SessionRecordResponse]:
        store = SessionStore(self._settings.db_path)
        await store.initialize()
        try:
            records = await store.list_sessions()
            return [SessionRecordResponse.model_validate(record.model_dump()) for record in records]
        finally:
            await store.close()

    async def shutdown(self) -> None:
        engines = list(self._engines.values())
        self._engines.clear()
        for engine in engines:
            with suppress(Exception):
                await engine.shutdown()


def _session_summary(engine: SessionEngine) -> SessionSummaryResponse:
    return SessionSummaryResponse(
        session_id=engine.session_id,
        model_name=engine.model_name,
        sub_agent_mode_enabled=engine.sub_agent_mode_enabled,
        max_sub_agents=engine.get_subagent_report().get("max_sub_agents", 0),
        available_models=engine.available_models,
        ws_path=_DEFAULT_WS_PATH_TEMPLATE.format(session_id=engine.session_id),
    )


def _worker_snapshot_key(report: dict[str, Any]) -> str:
    workers_raw = report.get("workers", [])
    if not isinstance(workers_raw, list):
        workers_raw = []
    workers = [worker for worker in workers_raw if isinstance(worker, dict)]
    payload = {
        "active_count": int(report.get("active_count", 0) or 0),
        "max_sub_agents": int(report.get("max_sub_agents", 0) or 0),
        "workers": workers,
    }
    return json.dumps(payload, sort_keys=True, default=str)


def create_api_app(settings: Settings, registry: CommandRegistry) -> FastAPI:
    manager = EngineManager(settings, registry)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            await manager.shutdown()

    app = FastAPI(title="LLC API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.api_allowed_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
        return _session_summary(engine)

    @app.get("/api/sessions/{session_id}", response_model=SessionSummaryResponse)
    async def get_session(session_id: str) -> SessionSummaryResponse:
        engine = await manager.get_or_create(session_id)
        return _session_summary(engine)

    @app.post("/api/sessions/{session_id}/restore", response_model=RestoreSessionResponse)
    async def restore_session(session_id: str) -> RestoreSessionResponse:
        engine = await manager.get_or_create(session_id)
        events: list[dict[str, Any]] = []
        async for event in engine.restore_session(session_id):
            events.append(event.model_dump(mode="json"))
        return RestoreSessionResponse(session=_session_summary(engine), events=events)

    @app.get("/api/sessions/{session_id}/subagents")
    async def get_subagent_report(session_id: str) -> dict[str, Any]:
        engine = await manager.get_or_create(session_id)
        return engine.get_subagent_report(include_all=True)

    @app.websocket("/api/ws/{session_id}")
    async def session_ws(websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        engine = await manager.get_or_create(session_id)
        send_lock = asyncio.Lock()

        async def send_event(event: Event) -> None:
            async with send_lock:
                await websocket.send_json(event.model_dump(mode="json"))

        async def send_error(message: str) -> None:
            await send_event(ErrorOccurred(message=message))

        async def stream_subagent_status() -> None:
            last_snapshot = ""
            while True:
                report = engine.get_subagent_report(include_all=True)
                snapshot = _worker_snapshot_key(report)
                if snapshot != last_snapshot:
                    last_snapshot = snapshot
                    workers_raw = report.get("workers", [])
                    workers = (
                        [worker for worker in workers_raw if isinstance(worker, dict)]
                        if isinstance(workers_raw, list)
                        else []
                    )
                    status_event = SubagentStatusUpdate(
                        workers=workers,
                        active_count=max(int(report.get("active_count", 0) or 0), 0),
                        max_sub_agents=max(int(report.get("max_sub_agents", 0) or 0), 0),
                    )
                    await send_event(status_event)
                await asyncio.sleep(settings.api_subagent_report_interval_s)

        poll_task = asyncio.create_task(stream_subagent_status())
        try:
            while True:
                raw_payload = await websocket.receive_json()
                try:
                    incoming = WebSocketMessage.model_validate(raw_payload)
                except ValidationError as exc:
                    await send_error(f"Invalid WebSocket payload: {exc.errors()!s}")
                    continue

                if incoming.action == "send_message":
                    text = incoming.text.strip()
                    if not text:
                        await send_error("Message text cannot be empty.")
                        continue
                    async for event in engine.send_message(text):
                        await send_event(event)
                    continue

                if incoming.action == "interrupt":
                    async for event in engine.interrupt():
                        await send_event(event)
                    continue

                if incoming.action == "restore_session":
                    target_session = (incoming.session_id or "").strip()
                    if not target_session:
                        await send_error("`session_id` is required for restore_session.")
                        continue
                    async for event in engine.restore_session(target_session):
                        await send_event(event)
                    continue
        except WebSocketDisconnect:
            return
        finally:
            poll_task.cancel()
            with suppress(asyncio.CancelledError):
                await poll_task

    return app

