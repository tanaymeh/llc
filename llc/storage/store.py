from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import aiosqlite

from llc.storage.models import (
    ConversationEventRecord,
    ConversationMessageRecord,
    ConversationRecord,
    ConversationStateSnapshotRecord,
    ConversationUsageRecord,
)
from llc.storage.schema import (
    INDEX_STATEMENTS,
    LEGACY_INDEX_STATEMENTS,
    LEGACY_SCHEMA_STATEMENTS,
    SCHEMA_STATEMENTS,
)

_ORCHESTRATOR_VISIBLE_STATE_KIND = "orchestrator_visible"


class ConversationStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._connection: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()
        self._event_write_lock = asyncio.Lock()

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
            for statement in LEGACY_SCHEMA_STATEMENTS:
                await connection.execute(statement)
            for statement in SCHEMA_STATEMENTS:
                await connection.execute(statement)
            for statement in LEGACY_INDEX_STATEMENTS:
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

    async def create_conversation(self, record: ConversationRecord) -> None:
        conn = await self._conn()
        payload = record.model_dump()
        await conn.execute(
            """
            INSERT INTO conversations (
                id,
                title,
                model_name,
                sub_agent_mode,
                created_at,
                updated_at,
                legacy_session_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title=excluded.title,
                model_name=excluded.model_name,
                sub_agent_mode=excluded.sub_agent_mode,
                updated_at=excluded.updated_at,
                legacy_session_id=excluded.legacy_session_id
            """,
            (
                payload["id"],
                payload["title"],
                payload["model_name"],
                int(bool(payload["sub_agent_mode"])),
                float(payload["created_at"]),
                float(payload["updated_at"]),
                payload["legacy_session_id"],
            ),
        )
        await conn.commit()

    async def ensure_conversation(
        self,
        conversation_id: str,
        *,
        model_name: str,
        sub_agent_mode: bool,
        title: str = "",
        created_at: float,
        updated_at: float,
    ) -> ConversationRecord:
        canonical = await self.get_canonical_conversation(conversation_id)
        if canonical is not None:
            return canonical

        legacy = await self.get_legacy_session(conversation_id)
        if legacy is not None:
            record = ConversationRecord(
                id=legacy.id,
                title=legacy.title,
                model_name=legacy.model_name,
                sub_agent_mode=legacy.sub_agent_mode,
                created_at=legacy.created_at,
                updated_at=max(legacy.updated_at, updated_at),
                legacy_session_id=legacy.id,
            )
            await self.create_conversation(record)
            return record

        record = ConversationRecord(
            id=conversation_id,
            title=title,
            model_name=model_name,
            sub_agent_mode=sub_agent_mode,
            created_at=created_at,
            updated_at=updated_at,
            legacy_session_id="",
        )
        await self.create_conversation(record)
        return record

    async def update_conversation(self, conversation_id: str, **fields: Any) -> None:
        conn = await self._conn()
        allowed = {
            "title",
            "model_name",
            "sub_agent_mode",
            "updated_at",
            "legacy_session_id",
        }
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
        values.append(conversation_id)

        sql = f"UPDATE conversations SET {', '.join(parts)} WHERE id = ?"
        await conn.execute(sql, tuple(values))
        await conn.commit()

    async def update_conversation_title(
        self,
        conversation_id: str,
        title: str,
    ) -> None:
        conn = await self._conn()
        await conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? "
            "WHERE id = ? AND (title IS NULL OR title = '')",
            (title, time.time(), conversation_id),
        )
        await conn.commit()

    async def get_canonical_conversation(
        self,
        conversation_id: str,
    ) -> ConversationRecord | None:
        conn = await self._conn()
        async with conn.execute(
            """
            SELECT
                id,
                title,
                model_name,
                sub_agent_mode,
                created_at,
                updated_at,
                legacy_session_id
            FROM conversations
            WHERE id = ?
            LIMIT 1
            """,
            (conversation_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return ConversationRecord.model_validate(_conversation_row_to_dict(row))

    async def get_conversation(self, conversation_id: str) -> ConversationRecord | None:
        canonical = await self.get_canonical_conversation(conversation_id)
        if canonical is not None:
            return canonical
        return await self.get_legacy_session(conversation_id)

    async def list_conversations(self, *, limit: int = 200) -> list[ConversationRecord]:
        conn = await self._conn()
        async with conn.execute(
            """
            SELECT
                id,
                title,
                model_name,
                sub_agent_mode,
                created_at,
                updated_at,
                legacy_session_id
            FROM conversations
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (max(int(limit), 1),),
        ) as cursor:
            rows = await cursor.fetchall()
        mapped = [ConversationRecord.model_validate(_conversation_row_to_dict(row)) for row in rows]

        legacy = await self.list_legacy_sessions(limit=limit)
        seen_ids = {record.id for record in mapped}
        combined = [*mapped, *(record for record in legacy if record.id not in seen_ids)]
        combined.sort(key=lambda record: record.updated_at, reverse=True)
        return combined[: max(int(limit), 1)]

    async def save_conversation_message(self, record: ConversationMessageRecord) -> None:
        conn = await self._conn()
        payload = record.model_dump()
        await conn.execute(
            """
            INSERT INTO conversation_messages (
                id,
                conversation_id,
                actor_kind,
                actor_id,
                turn_id,
                message_kind,
                role,
                content_json,
                tool_call_id,
                langfuse_trace_id,
                visible_to_orchestrator,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["id"],
                payload["conversation_id"],
                payload["actor_kind"],
                payload["actor_id"],
                payload["turn_id"],
                payload["message_kind"],
                payload["role"],
                payload["content_json"],
                payload["tool_call_id"],
                payload["langfuse_trace_id"],
                int(bool(payload["visible_to_orchestrator"])),
                float(payload["created_at"]),
            ),
        )
        await conn.commit()

    async def list_conversation_messages(
        self,
        conversation_id: str,
        *,
        actor_id: str | None = None,
        visible_to_orchestrator: bool | None = None,
        limit: int | None = None,
    ) -> list[ConversationMessageRecord]:
        conn = await self._conn()
        clauses = ["conversation_id = ?"]
        params: list[Any] = [conversation_id]
        if actor_id is not None:
            clauses.append("actor_id = ?")
            params.append(actor_id)
        if visible_to_orchestrator is not None:
            clauses.append("visible_to_orchestrator = ?")
            params.append(int(bool(visible_to_orchestrator)))
        sql = (
            """
            SELECT
                id,
                conversation_id,
                actor_kind,
                actor_id,
                turn_id,
                message_kind,
                role,
                content_json,
                tool_call_id,
                langfuse_trace_id,
                visible_to_orchestrator,
                created_at
            FROM conversation_messages
            WHERE """
            + " AND ".join(clauses)
            + """
            ORDER BY created_at ASC
            """
        )
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(int(limit), 1))
        async with conn.execute(sql, tuple(params)) as cursor:
            rows = await cursor.fetchall()
        return [
            ConversationMessageRecord.model_validate(_conversation_message_row_to_dict(row))
            for row in rows
        ]

    async def record_conversation_usage(self, record: ConversationUsageRecord) -> None:
        conn = await self._conn()
        payload = record.model_dump()
        await conn.execute(
            """
            INSERT INTO conversation_usage (
                conversation_id,
                actor_kind,
                actor_id,
                model_name,
                input_tokens,
                output_tokens,
                cost,
                recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["conversation_id"],
                payload["actor_kind"],
                payload["actor_id"],
                payload["model_name"],
                max(int(payload["input_tokens"]), 0),
                max(int(payload["output_tokens"]), 0),
                max(float(payload["cost"]), 0.0),
                float(payload["recorded_at"]),
            ),
        )
        await conn.commit()

    async def aggregate_conversation_usage(
        self,
        conversation_id: str,
    ) -> dict[str, dict[str, Any]]:
        conn = await self._conn()
        async with conn.execute(
            """
            SELECT
                actor_kind,
                model_name,
                SUM(input_tokens) AS input_tokens,
                SUM(output_tokens) AS output_tokens,
                SUM(cost) AS cost
            FROM conversation_usage
            WHERE conversation_id = ?
            GROUP BY actor_kind, model_name
            """,
            (conversation_id,),
        ) as cursor:
            rows = await cursor.fetchall()

        aggregates: dict[str, dict[str, Any]] = {}
        for row in rows:
            actor_kind = str(row["actor_kind"] or "").strip() or "orchestrator"
            model_name = str(row["model_name"] or "").strip()
            actor_bucket = aggregates.setdefault(
                actor_kind,
                {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost": 0.0,
                    "by_model": {},
                },
            )
            input_tokens = max(int(row["input_tokens"] or 0), 0)
            output_tokens = max(int(row["output_tokens"] or 0), 0)
            cost = max(float(row["cost"] or 0.0), 0.0)
            actor_bucket["input_tokens"] += input_tokens
            actor_bucket["output_tokens"] += output_tokens
            actor_bucket["cost"] += cost
            if model_name:
                actor_bucket["by_model"][model_name] = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost": cost,
                }
        return aggregates

    async def save_conversation_event(self, record: ConversationEventRecord) -> ConversationEventRecord:
        conn = await self._conn()
        payload = record.model_dump()
        async with self._event_write_lock:
            sequence_no = payload["sequence_no"]
            if sequence_no is None:
                sequence_no = await self._next_sequence_no(
                    conn,
                    conversation_id=payload["conversation_id"],
                )
            await conn.execute(
                """
                INSERT INTO conversation_events (
                    conversation_id,
                    sequence_no,
                    actor_kind,
                    actor_id,
                    turn_id,
                    event_type,
                    payload_json,
                    langfuse_trace_id,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["conversation_id"],
                    int(sequence_no),
                    payload["actor_kind"],
                    payload["actor_id"],
                    payload["turn_id"],
                    payload["event_type"],
                    payload["payload_json"],
                    payload["langfuse_trace_id"],
                    float(payload["created_at"]),
                ),
            )
            await conn.commit()
        return record.model_copy(update={"sequence_no": int(sequence_no)})

    async def list_conversation_events(
        self,
        conversation_id: str,
        *,
        actor_id: str | None = None,
        limit: int | None = None,
    ) -> list[ConversationEventRecord]:
        conn = await self._conn()
        clauses = ["conversation_id = ?"]
        params: list[Any] = [conversation_id]
        if actor_id is not None:
            clauses.append("actor_id = ?")
            params.append(actor_id)
        sql = (
            """
            SELECT
                id,
                conversation_id,
                sequence_no,
                actor_kind,
                actor_id,
                turn_id,
                event_type,
                payload_json,
                langfuse_trace_id,
                created_at
            FROM conversation_events
            WHERE """
            + " AND ".join(clauses)
            + """
            ORDER BY sequence_no ASC
            """
        )
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(int(limit), 1))
        async with conn.execute(sql, tuple(params)) as cursor:
            rows = await cursor.fetchall()
        return [
            ConversationEventRecord.model_validate(dict(row))
            for row in rows
        ]

    async def save_state_snapshot(self, record: ConversationStateSnapshotRecord) -> None:
        conn = await self._conn()
        payload = record.model_dump()
        await conn.execute(
            """
            INSERT INTO conversation_state_snapshots (
                conversation_id,
                state_kind,
                payload_json,
                updated_at
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(conversation_id, state_kind) DO UPDATE SET
                payload_json=excluded.payload_json,
                updated_at=excluded.updated_at
            """,
            (
                payload["conversation_id"],
                payload["state_kind"],
                payload["payload_json"],
                float(payload["updated_at"]),
            ),
        )
        await conn.commit()

    async def get_state_snapshot(
        self,
        conversation_id: str,
        *,
        state_kind: str = _ORCHESTRATOR_VISIBLE_STATE_KIND,
    ) -> ConversationStateSnapshotRecord | None:
        conn = await self._conn()
        async with conn.execute(
            """
            SELECT conversation_id, state_kind, payload_json, updated_at
            FROM conversation_state_snapshots
            WHERE conversation_id = ? AND state_kind = ?
            LIMIT 1
            """,
            (conversation_id, state_kind),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return ConversationStateSnapshotRecord.model_validate(dict(row))

    async def get_legacy_session(self, session_id: str) -> ConversationRecord | None:
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
        payload = dict(row)
        payload["sub_agent_mode"] = bool(payload.get("sub_agent_mode", 0))
        payload["legacy_session_id"] = payload["id"]
        return ConversationRecord.model_validate(payload)

    async def list_legacy_sessions(self, *, limit: int = 200) -> list[ConversationRecord]:
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
        records: list[ConversationRecord] = []
        for row in rows:
            payload = dict(row)
            payload["sub_agent_mode"] = bool(payload.get("sub_agent_mode", 0))
            payload["legacy_session_id"] = payload["id"]
            records.append(ConversationRecord.model_validate(payload))
        return records

    async def get_legacy_messages(
        self,
        session_id: str,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
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
        return [dict(row) for row in rows]

    async def _next_sequence_no(
        self,
        conn: aiosqlite.Connection,
        *,
        conversation_id: str,
    ) -> int:
        async with conn.execute(
            """
            SELECT COALESCE(MAX(sequence_no), 0) + 1
            FROM conversation_events
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return 1
        return max(int(row[0] or 1), 1)

    async def _conn(self) -> aiosqlite.Connection:
        if self._connection is None:
            await self.initialize()
        if self._connection is None:
            raise RuntimeError("Database connection is not initialized")
        return self._connection

    # Compatibility wrappers while the service layer still uses session terminology.
    async def create_session(self, record: ConversationRecord) -> None:
        await self.create_conversation(record)

    async def update_session(self, session_id: str, **fields: Any) -> None:
        await self.update_conversation(session_id, **fields)

    async def get_session(self, session_id: str) -> ConversationRecord | None:
        return await self.get_conversation(session_id)

    async def list_sessions(self, *, limit: int = 200) -> list[ConversationRecord]:
        return await self.list_conversations(limit=limit)

    async def save_message(self, record: ConversationMessageRecord) -> None:
        await self.save_conversation_message(record)

    async def get_messages(
        self,
        session_id: str,
        *,
        limit: int | None = None,
    ) -> list[ConversationMessageRecord]:
        return await self.list_conversation_messages(session_id, limit=limit)

    async def record_usage(self, record: ConversationUsageRecord) -> None:
        await self.record_conversation_usage(record)

    async def save_event(self, record: ConversationEventRecord) -> ConversationEventRecord:
        return await self.save_conversation_event(record)


def _conversation_row_to_dict(row: aiosqlite.Row) -> dict[str, Any]:
    payload = dict(row)
    payload["sub_agent_mode"] = bool(payload.get("sub_agent_mode", 0))
    return payload


def _conversation_message_row_to_dict(row: aiosqlite.Row) -> dict[str, Any]:
    payload = dict(row)
    payload["visible_to_orchestrator"] = bool(payload.get("visible_to_orchestrator", 0))
    return payload


SessionStore = ConversationStore
