from __future__ import annotations

SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        title TEXT DEFAULT '',
        model_name TEXT NOT NULL,
        sub_agent_mode INTEGER NOT NULL DEFAULT 0,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        tool_calls_json TEXT NOT NULL DEFAULT '[]',
        usage_input_tokens INTEGER NOT NULL DEFAULT 0,
        usage_output_tokens INTEGER NOT NULL DEFAULT 0,
        created_at REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS token_usage (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        model_name TEXT NOT NULL,
        input_tokens INTEGER NOT NULL DEFAULT 0,
        output_tokens INTEGER NOT NULL DEFAULT 0,
        cost REAL NOT NULL DEFAULT 0.0,
        recorded_at REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        turn_id TEXT NOT NULL DEFAULT '',
        event_type TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at REAL NOT NULL
    )
    """,
)

INDEX_STATEMENTS: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_messages_session_created_at ON messages(session_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_token_usage_session_recorded_at ON token_usage(session_id, recorded_at)",
    "CREATE INDEX IF NOT EXISTS idx_events_session_created_at ON events(session_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_events_turn ON events(turn_id)",
)

