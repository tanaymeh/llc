from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import aiosqlite

from llc.storage.models import EventRecord, MessageRecord, SessionRecord, TokenUsageRecord
from llc.storage.schema import INDEX_STATEMENTS, SCHEMA_STATEMENTS


class SessionStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._connection: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    @property
    def db_path(self) -> Path:
        return self._db_path

    async def initialize(self) -> None:
        async with self._lock:
            if self._connection is not None:
                return

            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            connection = await aiosqlite.connect(str(self._db_path))
            connection.row_factory = aiosqlite.Row
            await connection.execute("PRAGMA foreign_keys = ON")
            for statement in SCHEMA_STATEMENTS:
                await connection.execute(statement)
            for statement in INDEX_STATEMENTS:
                await connection.execute(statement)
            await connection.commit()
            self._connection = connection

    async def close(self) -> None:
        async with self._lock:
            if self._connection is None:
                return
            await self._connection.close()
            self._connection = None

    async def create_session(self, record: SessionRecord) -> None:
        conn = await self._conn()
        payload = record.model_dump()
        await conn.execute(
            """
            INSERT INTO sessions (
                id, title, model_name, sub_agent_mode, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title=excluded.title,
                model_name=excluded.model_name,
                sub_agent_mode=excluded.sub_agent_mode,
                updated_at=excluded.updated_at
            """,
            (
                payload["id"],
                payload["title"],
                payload["model_name"],
                int(bool(payload["sub_agent_mode"])),
                float(payload["created_at"]),
                float(payload["updated_at"]),
            ),
        )
        await conn.commit()

    async def update_session(self, session_id: str, **fields: Any) -> None:
        conn = await self._conn()
        allowed = {"title", "model_name", "sub_agent_mode", "updated_at"}
        updates = {key: value for key, value in fields.items() if key in allowed}
        if not updates:
            return

        parts: list[str] = []
        values: list[Any] = []
        for key, value in updates.items():
            parts.append(f"{key} = ?")
            if key == "sub_agent_mode":
                values.append(int(bool(value)))
            else:
                values.append(value)
        values.append(session_id)

        sql = f"UPDATE sessions SET {', '.join(parts)} WHERE id = ?"
        await conn.execute(sql, tuple(values))
        await conn.commit()

    async def get_session(self, session_id: str) -> SessionRecord | None:
        conn = await self._conn()
        async with conn.execute(
            """
            SELECT id, title, model_name, sub_agent_mode, created_at, updated_at
            FROM sessions
            WHERE id = ?
            LIMIT 1
            """,
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return SessionRecord.model_validate(_session_row_to_dict(row))

    async def list_sessions(self, *, limit: int = 200) -> list[SessionRecord]:
        conn = await self._conn()
        async with conn.execute(
            """
            SELECT id, title, model_name, sub_agent_mode, created_at, updated_at
            FROM sessions
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (max(int(limit), 1),),
        ) as cursor:
            rows = await cursor.fetchall()
        return [SessionRecord.model_validate(_session_row_to_dict(row)) for row in rows]

    async def save_message(self, record: MessageRecord) -> None:
        conn = await self._conn()
        payload = record.model_dump()
        await conn.execute(
            """
            INSERT INTO messages (
                id,
                session_id,
                role,
                content,
                tool_calls_json,
                usage_input_tokens,
                usage_output_tokens,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["id"],
                payload["session_id"],
                payload["role"],
                payload["content"],
                payload["tool_calls_json"],
                max(int(payload["usage_input_tokens"]), 0),
                max(int(payload["usage_output_tokens"]), 0),
                float(payload["created_at"]),
            ),
        )
        await conn.commit()

    async def get_messages(self, session_id: str, *, limit: int | None = None) -> list[MessageRecord]:
        conn = await self._conn()
        if limit is None:
            query = """
                SELECT
                    id,
                    session_id,
                    role,
                    content,
                    tool_calls_json,
                    usage_input_tokens,
                    usage_output_tokens,
                    created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY created_at ASC
            """
            params: tuple[Any, ...] = (session_id,)
        else:
            query = """
                SELECT
                    id,
                    session_id,
                    role,
                    content,
                    tool_calls_json,
                    usage_input_tokens,
                    usage_output_tokens,
                    created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY created_at ASC
                LIMIT ?
            """
            params = (session_id, max(int(limit), 1))
        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
        return [MessageRecord.model_validate(dict(row)) for row in rows]

    async def record_usage(self, record: TokenUsageRecord) -> None:
        conn = await self._conn()
        payload = record.model_dump()
        await conn.execute(
            """
            INSERT INTO token_usage (
                session_id,
                model_name,
                input_tokens,
                output_tokens,
                cost,
                recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                payload["session_id"],
                payload["model_name"],
                max(int(payload["input_tokens"]), 0),
                max(int(payload["output_tokens"]), 0),
                max(float(payload["cost"]), 0.0),
                float(payload["recorded_at"]),
            ),
        )
        await conn.commit()

    async def save_event(self, record: EventRecord) -> None:
        conn = await self._conn()
        payload = record.model_dump()
        await conn.execute(
            """
            INSERT INTO events (
                session_id,
                turn_id,
                event_type,
                payload_json,
                created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                payload["session_id"],
                payload["turn_id"],
                payload["event_type"],
                payload["payload_json"],
                float(payload["created_at"]),
            ),
        )
        await conn.commit()

    async def _conn(self) -> aiosqlite.Connection:
        if self._connection is None:
            await self.initialize()
        if self._connection is None:
            raise RuntimeError("Database connection is not initialized")
        return self._connection


def _session_row_to_dict(row: aiosqlite.Row) -> dict[str, Any]:
    payload = dict(row)
    payload["sub_agent_mode"] = bool(payload.get("sub_agent_mode", 0))
    return payload

