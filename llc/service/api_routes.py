from __future__ import annotations

import json
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
from llc.storage.store import SessionStore


async def _conversation_store(manager: EngineManager) -> SessionStore:
    store = SessionStore(manager._settings.db_path)
    await store.initialize()
    return store


def _snapshot_transcript_records(
    conversation_id: str,
    payload_json: str,
    *,
    created_at: float,
) -> list[dict[str, Any]]:
    try:
        payload = json.loads(payload_json)
    except Exception:
        return []
    if not isinstance(payload, list):
        return []

    records: list[dict[str, Any]] = []
    role_by_class = {
        "HumanMessage": "user",
        "AIMessage": "assistant",
        "SystemMessage": "system",
        "ToolMessage": "tool",
    }
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            continue
        message_class = str(item.get("message_class", "") or "").strip()
        role = role_by_class.get(message_class)
        if role is None:
            continue
        raw_tool_calls = item.get("tool_calls", [])
        tool_calls = raw_tool_calls if isinstance(raw_tool_calls, list) else []
        content_payload = {
            "content": item.get("content", ""),
            "tool_calls": tool_calls,
            "usage_input_tokens": 0,
            "usage_output_tokens": 0,
        }
        records.append(
            {
                "id": f"snapshot-{conversation_id}-{index}",
                "conversation_id": conversation_id,
                "actor_kind": "orchestrator",
                "actor_id": "orchestrator",
                "turn_id": "",
                "message_kind": "checkpoint",
                "role": role,
                "content_json": json.dumps(content_payload, sort_keys=True, default=str),
                "tool_call_id": str(item.get("tool_call_id", "") or ""),
                "langfuse_trace_id": "",
                "visible_to_orchestrator": True,
                "created_at": created_at,
            }
        )
    return records


def register_api_routes(app: FastAPI, manager: EngineManager) -> None:
    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"ok": True}

    @app.get("/api/sessions", response_model=list[SessionRecordResponse])
    @app.get("/api/conversations", response_model=list[SessionRecordResponse])
    async def list_sessions() -> list[SessionRecordResponse]:
        return await manager.list_sessions()

    @app.post("/api/sessions", response_model=SessionSummaryResponse)
    @app.post("/api/conversations", response_model=SessionSummaryResponse)
    async def create_session(
        payload: CreateSessionRequest | None = None,
    ) -> SessionSummaryResponse:
        engine = await manager.get_or_create(
            payload.requested_conversation_id if payload else None
        )
        return session_summary(engine)

    @app.get("/api/sessions/{session_id}", response_model=SessionSummaryResponse)
    @app.get("/api/conversations/{session_id}", response_model=SessionSummaryResponse)
    async def get_session(session_id: str) -> SessionSummaryResponse:
        engine = await manager.get_or_create(session_id)
        return session_summary(engine)

    @app.post("/api/sessions/{session_id}/restore", response_model=RestoreSessionResponse)
    @app.post(
        "/api/conversations/{session_id}/restore",
        response_model=RestoreSessionResponse,
    )
    async def restore_session(session_id: str) -> RestoreSessionResponse:
        engine = await manager.get_or_create(session_id)
        events: list[dict[str, Any]] = []
        async for event in engine.restore_session(session_id):
            events.append(event.model_dump(mode="json"))
        return RestoreSessionResponse(session=session_summary(engine), events=events)

    @app.get("/api/sessions/{session_id}/subagents")
    @app.get("/api/conversations/{session_id}/subagents")
    async def get_subagent_report(session_id: str) -> dict[str, Any]:
        engine = await manager.get_or_create(session_id)
        return engine.get_subagent_report(include_all=True)

    @app.get("/api/sessions/{conversation_id}/messages")
    @app.get("/api/conversations/{conversation_id}/messages")
    async def list_conversation_messages(
        conversation_id: str,
        actor_id: str | None = None,
        visible_to_orchestrator: bool | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        store = await _conversation_store(manager)
        try:
            records = await store.list_conversation_messages(
                conversation_id,
                actor_id=actor_id,
                visible_to_orchestrator=visible_to_orchestrator,
                limit=limit,
            )
            return [record.model_dump(mode="json") for record in records]
        finally:
            await store.close()

    @app.get("/api/sessions/{conversation_id}/events")
    @app.get("/api/conversations/{conversation_id}/events")
    async def list_conversation_events(
        conversation_id: str,
        actor_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        store = await _conversation_store(manager)
        try:
            records = await store.list_conversation_events(
                conversation_id,
                actor_id=actor_id,
                limit=limit,
            )
            return [record.model_dump(mode="json") for record in records]
        finally:
            await store.close()

    @app.get("/api/sessions/{conversation_id}/transcript")
    @app.get("/api/conversations/{conversation_id}/transcript")
    async def get_conversation_transcript(
        conversation_id: str,
        actor_id: str | None = None,
    ) -> list[dict[str, Any]]:
        store = await _conversation_store(manager)
        try:
            if actor_id is not None and actor_id.strip():
                records = await store.list_conversation_messages(
                    conversation_id,
                    actor_id=actor_id.strip(),
                )
                return [record.model_dump(mode="json") for record in records]

            records = await store.list_conversation_messages(
                conversation_id,
                visible_to_orchestrator=True,
            )
            if records:
                return [record.model_dump(mode="json") for record in records]

            snapshot = await store.get_state_snapshot(conversation_id)
            if snapshot is not None:
                snapshot_records = _snapshot_transcript_records(
                    conversation_id,
                    snapshot.payload_json,
                    created_at=float(snapshot.updated_at or 0.0),
                )
                if snapshot_records:
                    return snapshot_records

            return []
        finally:
            await store.close()
