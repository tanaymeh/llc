from __future__ import annotations

import json
import time
import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from llc.service.events import Event


async def persist_message(
    engine: Any,
    *,
    role: str,
    content: str,
    tool_calls: list[dict[str, Any]] | None = None,
    usage_input_tokens: int = 0,
    usage_output_tokens: int = 0,
) -> None:
    if engine._store is None:
        return
    from llc.storage.models import MessageRecord

    await engine._store.save_message(
        MessageRecord(
            id=f"msg-{uuid.uuid4().hex}",
            session_id=engine._session_id,
            role=role,
            content=content,
            tool_calls_json=json.dumps(tool_calls or [], sort_keys=True, default=str),
            usage_input_tokens=max(int(usage_input_tokens), 0),
            usage_output_tokens=max(int(usage_output_tokens), 0),
            created_at=time.time(),
        )
    )
    await engine._store.update_session(
        engine._session_id,
        model_name=engine._settings.model_name,
        sub_agent_mode=engine._settings.sub_agent_mode_enabled,
        updated_at=time.time(),
    )


async def persist_usage(
    engine: Any,
    *,
    input_tokens: int,
    output_tokens: int,
) -> None:
    if engine._store is None:
        return
    if input_tokens <= 0 and output_tokens <= 0:
        return
    from llc.storage.models import TokenUsageRecord

    await engine._store.record_usage(
        TokenUsageRecord(
            session_id=engine._session_id,
            model_name=engine._settings.model_name,
            input_tokens=max(input_tokens, 0),
            output_tokens=max(output_tokens, 0),
            cost=max(engine._turn_cost(input_tokens, output_tokens), 0.0),
            recorded_at=time.time(),
        )
    )


async def persist_event(engine: Any, event: Event, *, turn_id: str) -> None:
    if engine._store is None:
        return
    from llc.storage.models import EventRecord

    await engine._store.save_event(
        EventRecord(
            session_id=engine._session_id,
            turn_id=turn_id,
            event_type=event.type,
            payload_json=event.model_dump_json(),
            created_at=time.time(),
        )
    )


async def restore_messages_to_agent(engine: Any, records: list[Any]) -> None:
    if engine._agent is None:
        return

    restored: list[Any] = []
    for record in records:
        role = str(getattr(record, "role", "")).strip().lower()
        content = str(getattr(record, "content", ""))
        if role == "user":
            restored.append(HumanMessage(content=content))
            continue
        if role == "assistant":
            raw_tool_calls = getattr(record, "tool_calls_json", "[]")
            try:
                parsed = json.loads(raw_tool_calls)
            except Exception:
                parsed = []
            tool_calls = parsed if isinstance(parsed, list) else []
            restored.append(
                AIMessage(
                    content=content,
                    tool_calls=tool_calls,
                )
            )
            continue
        if role == "system":
            restored.append(SystemMessage(content=content))
            continue
        if role == "tool":
            restored.append(
                ToolMessage(
                    content=content,
                    tool_call_id=f"restored-{uuid.uuid4().hex[:8]}",
                )
            )

    if not restored:
        return

    await engine._agent.aupdate_state(
        {"configurable": {"thread_id": engine._thread_id}},
        {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *restored]},
        as_node="llm",
    )
