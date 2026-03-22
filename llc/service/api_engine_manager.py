from __future__ import annotations

import asyncio
from contextlib import suppress

from llc.commands import CommandRegistry
from llc.config import Settings
from llc.service.api_models import SessionRecordResponse
from llc.service.engine import SessionEngine
from llc.storage.store import SessionStore


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
                enable_langfuse_tracing=True,
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
