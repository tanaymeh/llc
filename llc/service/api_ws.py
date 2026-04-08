from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from llc.config import Settings
from llc.service.api_engine_manager import EngineManager
from llc.service.api_models import WebSocketMessage
from llc.service.events import ErrorOccurred, Event, SessionRestored, SubagentStatusUpdate


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


def register_ws_routes(app: FastAPI, manager: EngineManager, settings: Settings) -> None:
    @app.websocket("/api/ws/{conversation_id}")
    async def session_ws(websocket: WebSocket, conversation_id: str) -> None:
        await websocket.accept()
        engine = await manager.get_or_create(conversation_id)
        async_event_queue = engine.subscribe_async_events()
        send_lock = asyncio.Lock()
        active_turn_task: asyncio.Task[None] | None = None
        stream_tasks: set[asyncio.Task[None]] = set()

        async def send_event(event: Event) -> None:
            async with send_lock:
                await websocket.send_json(event.model_dump(mode="json"))

        async def send_error(message: str) -> None:
            await send_event(ErrorOccurred(message=message))

        async def stream_engine_events(source: Any) -> None:
            try:
                async for event in source:
                    await send_event(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                await send_error(str(exc))

        async def stream_async_engine_events() -> None:
            while True:
                event = await async_event_queue.get()
                await send_event(event)

        async def restart_engine_streams(target_conversation_id: str) -> None:
            nonlocal engine, async_event_queue, poll_task, async_events_task
            if target_conversation_id == engine.conversation_id:
                return
            engine.unsubscribe_async_events(async_event_queue)
            poll_task.cancel()
            with suppress(asyncio.CancelledError):
                await poll_task
            async_events_task.cancel()
            with suppress(asyncio.CancelledError):
                await async_events_task
            engine = await manager.get_or_create(target_conversation_id)
            async_event_queue = engine.subscribe_async_events()
            poll_task = asyncio.create_task(stream_subagent_status())
            async_events_task = asyncio.create_task(stream_async_engine_events())

        def track_stream_task(task: asyncio.Task[None], *, marks_active_turn: bool) -> None:
            nonlocal active_turn_task
            stream_tasks.add(task)
            if marks_active_turn:
                active_turn_task = task

            def _cleanup(done: asyncio.Task[None]) -> None:
                nonlocal active_turn_task
                stream_tasks.discard(done)
                if active_turn_task is done:
                    active_turn_task = None
                with suppress(asyncio.CancelledError, Exception):
                    done.result()

            task.add_done_callback(_cleanup)

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
        async_events_task = asyncio.create_task(stream_async_engine_events())
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
                    if active_turn_task is not None and not active_turn_task.done():
                        await send_error("A turn is already running.")
                        continue
                    task = asyncio.create_task(stream_engine_events(engine.send_message(text)))
                    track_stream_task(task, marks_active_turn=True)
                    continue

                if incoming.action == "interrupt":
                    task = asyncio.create_task(stream_engine_events(engine.interrupt()))
                    track_stream_task(task, marks_active_turn=False)
                    continue

                if incoming.action == "restore_session":
                    target_conversation = (
                        incoming.requested_conversation_id or ""
                    ).strip()
                    if not target_conversation:
                        await send_error(
                            "`conversation_id` (or legacy `session_id`) is required for restore_session."
                        )
                        continue
                    if active_turn_task is not None and not active_turn_task.done():
                        await send_error("A turn is already running.")
                        continue
                    if target_conversation != engine.conversation_id:
                        target_engine = await manager.get_or_create(target_conversation)
                        restored_events = [
                            event
                            async for event in target_engine.restore_session(
                                target_conversation
                            )
                        ]
                        for event in restored_events:
                            await send_event(event)
                        if any(
                            isinstance(event, SessionRestored)
                            for event in restored_events
                        ):
                            await restart_engine_streams(target_conversation)
                        continue
                    task = asyncio.create_task(
                        stream_engine_events(
                            engine.restore_session(target_conversation)
                        )
                    )
                    track_stream_task(task, marks_active_turn=True)
                    continue
        except WebSocketDisconnect:
            return
        finally:
            engine.unsubscribe_async_events(async_event_queue)
            poll_task.cancel()
            with suppress(asyncio.CancelledError):
                await poll_task
            async_events_task.cancel()
            with suppress(asyncio.CancelledError):
                await async_events_task
            for task in list(stream_tasks):
                task.cancel()
            for task in list(stream_tasks):
                with suppress(asyncio.CancelledError):
                    await task
