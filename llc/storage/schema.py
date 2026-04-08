from __future__ import annotations

LEGACY_SCHEMA_STATEMENTS: tuple[str, ...] = (
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

SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL DEFAULT '',
        model_name TEXT NOT NULL,
        sub_agent_mode INTEGER NOT NULL DEFAULT 0,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        legacy_session_id TEXT NOT NULL DEFAULT ''
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS conversation_messages (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        actor_kind TEXT NOT NULL,
        actor_id TEXT NOT NULL DEFAULT '',
        turn_id TEXT NOT NULL DEFAULT '',
        message_kind TEXT NOT NULL DEFAULT 'chat',
        role TEXT NOT NULL,
        content_json TEXT NOT NULL,
        tool_call_id TEXT NOT NULL DEFAULT '',
        langfuse_trace_id TEXT NOT NULL DEFAULT '',
        visible_to_orchestrator INTEGER NOT NULL DEFAULT 0,
        created_at REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS conversation_usage (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        actor_kind TEXT NOT NULL DEFAULT 'orchestrator',
        actor_id TEXT NOT NULL DEFAULT '',
        model_name TEXT NOT NULL,
        input_tokens INTEGER NOT NULL DEFAULT 0,
        output_tokens INTEGER NOT NULL DEFAULT 0,
        cost REAL NOT NULL DEFAULT 0.0,
        recorded_at REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS conversation_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        sequence_no INTEGER NOT NULL,
        actor_kind TEXT NOT NULL DEFAULT 'orchestrator',
        actor_id TEXT NOT NULL DEFAULT '',
        turn_id TEXT NOT NULL DEFAULT '',
        event_type TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        langfuse_trace_id TEXT NOT NULL DEFAULT '',
        created_at REAL NOT NULL,
        UNIQUE(conversation_id, sequence_no)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS conversation_state_snapshots (
        conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        state_kind TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        updated_at REAL NOT NULL,
        PRIMARY KEY (conversation_id, state_kind)
    )
    """,
)

LEGACY_INDEX_STATEMENTS: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_messages_session_created_at ON messages(session_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_token_usage_session_recorded_at ON token_usage(session_id, recorded_at)",
    "CREATE INDEX IF NOT EXISTS idx_events_session_created_at ON events(session_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_events_turn ON events(turn_id)",
)

INDEX_STATEMENTS: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_conversation_messages_created_at ON conversation_messages(conversation_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_conversation_messages_actor ON conversation_messages(conversation_id, actor_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_conversation_messages_visible ON conversation_messages(conversation_id, visible_to_orchestrator, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_conversation_usage_recorded_at ON conversation_usage(conversation_id, recorded_at)",
    "CREATE INDEX IF NOT EXISTS idx_conversation_usage_actor ON conversation_usage(conversation_id, actor_id, recorded_at)",
    "CREATE INDEX IF NOT EXISTS idx_conversation_events_created_at ON conversation_events(conversation_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_conversation_events_actor ON conversation_events(conversation_id, actor_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_conversation_events_turn ON conversation_events(turn_id)",
)
