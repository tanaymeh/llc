from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from llc.commands import CommandRegistry
from llc.config import Settings
from llc.observability import shutdown as shutdown_langfuse
from llc.service.api_engine_manager import EngineManager
from llc.service.api_models import (
    CreateSessionRequest,
    RestoreSessionResponse,
    SessionRecordResponse,
    SessionSummaryResponse,
    WebSocketMessage,
)
from llc.service.api_routes import register_api_routes
from llc.service.api_ws import register_ws_routes


__all__ = [
    "CreateSessionRequest",
    "RestoreSessionResponse",
    "SessionRecordResponse",
    "SessionSummaryResponse",
    "WebSocketMessage",
    "EngineManager",
    "create_api_app",
]


def create_api_app(settings: Settings, registry: CommandRegistry) -> FastAPI:
    manager = EngineManager(settings, registry)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            await manager.shutdown()
            shutdown_langfuse()

    app = FastAPI(title="LLC API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.api_allowed_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_api_routes(app, manager)
    register_ws_routes(app, manager, settings)
    return app
