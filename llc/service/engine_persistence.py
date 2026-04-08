from __future__ import annotations

import json
import time
import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from llc.observability import current_context, current_trace_id
from llc.service.events import Event

_ORCHESTRATOR_ACTOR_ID = "orchestrator"
_ORCHESTRATOR_VISIBLE_STATE_KIND = "orchestrator_visible"


async def persist_message(
    engine: Any,
    *,
    role: str,
    content: str,
    tool_calls: list[dict[str, Any]] | None = None,
    usage_input_tokens: int = 0,
    usage_output_tokens: int = 0,
    actor_kind: str = "orchestrator",
    actor_id: str = _ORCHESTRATOR_ACTOR_ID,
    turn_id: str = "",
    message_kind: str = "chat",
    tool_call_id: str = "",
    visible_to_orchestrator: bool = False,
    langfuse_trace_id: str = "",
) -> None:
    if engine._store is None:
        return
    from llc.storage.models import MessageRecord

    trace_id = (langfuse_trace_id or current_trace_id()).strip()
    message_payload = {
        "content": content,
        "tool_calls": tool_calls or [],
        "usage_input_tokens": max(int(usage_input_tokens), 0),
        "usage_output_tokens": max(int(usage_output_tokens), 0),
    }
    await engine._store.save_message(
        MessageRecord(
            id=f"msg-{uuid.uuid4().hex}",
            conversation_id=engine.conversation_id,
            actor_kind=actor_kind,
            actor_id=(actor_id or _ORCHESTRATOR_ACTOR_ID).strip(),
            turn_id=turn_id.strip(),
            message_kind=message_kind,
            role=role,
            content_json=json.dumps(message_payload, sort_keys=True, default=str),
            tool_call_id=tool_call_id.strip(),
            langfuse_trace_id=trace_id,
            visible_to_orchestrator=visible_to_orchestrator,
            created_at=time.time(),
        )
    )
    await engine._store.update_session(
        engine.conversation_id,
        model_name=engine._settings.model_name,
        sub_agent_mode=engine._settings.sub_agent_mode_enabled,
        updated_at=time.time(),
    )


async def persist_usage(
    engine: Any,
    *,
    input_tokens: int,
    output_tokens: int,
    actor_kind: str = "orchestrator",
    actor_id: str = _ORCHESTRATOR_ACTOR_ID,
) -> None:
    if engine._store is None:
        return
    if input_tokens <= 0 and output_tokens <= 0:
        return
    from llc.storage.models import TokenUsageRecord

    await engine._store.record_usage(
        TokenUsageRecord(
            conversation_id=engine.conversation_id,
            actor_kind=actor_kind,
            actor_id=(actor_id or _ORCHESTRATOR_ACTOR_ID).strip(),
            model_name=engine._settings.model_name,
            input_tokens=max(input_tokens, 0),
            output_tokens=max(output_tokens, 0),
            cost=max(engine._turn_cost(input_tokens, output_tokens), 0.0),
            recorded_at=time.time(),
        )
    )


async def persist_event(
    engine: Any,
    event: Event | None = None,
    *,
    turn_id: str,
    actor_kind: str | None = None,
    actor_id: str | None = None,
    event_type: str | None = None,
    payload: dict[str, Any] | None = None,
    langfuse_trace_id: str = "",
) -> None:
    if engine._store is None:
        return
    from llc.storage.models import EventRecord

    ctx = current_context()
    resolved_actor_kind = (actor_kind or getattr(ctx, "actor_kind", "") or "orchestrator").strip()
    resolved_actor_id = (actor_id or getattr(ctx, "actor_id", "") or _ORCHESTRATOR_ACTOR_ID).strip()
    resolved_trace_id = (langfuse_trace_id or current_trace_id()).strip()
    if event is not None:
        resolved_event_type = event.type
        payload_json = event.model_dump_json()
    else:
        resolved_event_type = (event_type or "").strip()
        payload_json = json.dumps(payload or {}, sort_keys=True, default=str)
    if not resolved_event_type:
        return

    await engine._store.save_event(
        EventRecord(
            conversation_id=engine.conversation_id,
            turn_id=turn_id.strip(),
            actor_kind=resolved_actor_kind,
            actor_id=resolved_actor_id,
            event_type=resolved_event_type,
            payload_json=payload_json,
            langfuse_trace_id=resolved_trace_id,
            created_at=time.time(),
        )
    )


async def persist_orchestrator_snapshot(engine: Any) -> None:
    if engine._store is None or engine._agent is None:
        return
    from llc.storage.models import ConversationStateSnapshotRecord

    messages = await _history_messages(engine._agent, engine._thread_id)
    payload = [_serialize_message(message) for message in messages]
    await engine._store.save_state_snapshot(
        ConversationStateSnapshotRecord(
            conversation_id=engine.conversation_id,
            state_kind=_ORCHESTRATOR_VISIBLE_STATE_KIND,
            payload_json=json.dumps(payload, sort_keys=True, default=str),
            updated_at=time.time(),
        )
    )


async def restore_messages_to_agent(engine: Any, records: list[Any] | None = None) -> None:
    if engine._agent is None or engine._store is None:
        return

    snapshot = await engine._store.get_state_snapshot(
        engine.conversation_id,
        state_kind=_ORCHESTRATOR_VISIBLE_STATE_KIND,
    )
    if snapshot is not None:
        restored = _deserialize_snapshot_payload(snapshot.payload_json)
        if restored:
            await _apply_restored_messages(engine, restored)
            return

    visible_messages = await engine._store.list_conversation_messages(
        engine.conversation_id,
        visible_to_orchestrator=True,
    )
    if visible_messages:
        restored = _conversation_records_to_messages(visible_messages)
        if restored:
            await _apply_restored_messages(engine, restored)
            return

    legacy_records = records
    if legacy_records is None:
        legacy_records = await engine._store.get_legacy_messages(engine.conversation_id)
    restored = _legacy_records_to_messages(legacy_records)
    if restored:
        await _apply_restored_messages(engine, restored)


async def _history_messages(agent: Any, thread_id: str) -> list[Any]:
    state = await agent.aget_state({"configurable": {"thread_id": thread_id}})
    values = getattr(state, "values", {})
    messages = values.get("messages", [])
    if not isinstance(messages, list):
        return []
    return list(messages)


def _serialize_message(message: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "message_class": message.__class__.__name__,
        "content": getattr(message, "content", ""),
        "additional_kwargs": getattr(message, "additional_kwargs", {}) or {},
    }
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        payload["tool_calls"] = tool_calls
    tool_call_id = getattr(message, "tool_call_id", "")
    if tool_call_id:
        payload["tool_call_id"] = tool_call_id
    return payload


def _deserialize_snapshot_payload(payload_json: str) -> list[Any]:
    try:
        payload = json.loads(payload_json)
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    restored: list[Any] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        restored_message = _deserialize_snapshot_message(item)
        if restored_message is not None:
            restored.append(restored_message)
    return restored


def _deserialize_snapshot_message(payload: dict[str, Any]) -> Any | None:
    message_class = str(payload.get("message_class", "")).strip()
    content = payload.get("content", "")
    additional_kwargs = payload.get("additional_kwargs", {}) or {}
    if message_class == "HumanMessage":
        return HumanMessage(content=content)
    if message_class == "AIMessage":
        return AIMessage(
            content=content,
            tool_calls=payload.get("tool_calls", []) if isinstance(payload.get("tool_calls"), list) else [],
            additional_kwargs=additional_kwargs if isinstance(additional_kwargs, dict) else {},
        )
    if message_class == "SystemMessage":
        return SystemMessage(content=content)
    if message_class == "ToolMessage":
        tool_call_id = str(payload.get("tool_call_id", "") or "").strip() or f"restored-{uuid.uuid4().hex[:8]}"
        return ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
            additional_kwargs=additional_kwargs if isinstance(additional_kwargs, dict) else {},
        )
    return None


def _conversation_records_to_messages(records: list[Any]) -> list[Any]:
    restored: list[Any] = []
    for record in records:
        role = str(getattr(record, "role", "")).strip().lower()
        raw_content_json = str(getattr(record, "content_json", "") or "")
        try:
            payload = json.loads(raw_content_json)
        except Exception:
            payload = {"content": raw_content_json}
        content = payload.get("content", "")
        if role == "user":
            restored.append(HumanMessage(content=content))
            continue
        if role == "assistant":
            raw_tool_calls = payload.get("tool_calls", [])
            tool_calls = raw_tool_calls if isinstance(raw_tool_calls, list) else []
            restored.append(AIMessage(content=content, tool_calls=tool_calls))
            continue
        if role == "system":
            restored.append(SystemMessage(content=content))
            continue
        if role == "tool":
            restored.append(
                ToolMessage(
                    content=content,
                    tool_call_id=str(getattr(record, "tool_call_id", "") or "").strip()
                    or f"restored-{uuid.uuid4().hex[:8]}",
                )
            )
    return restored


def _legacy_records_to_messages(records: list[Any]) -> list[Any]:
    restored: list[Any] = []
    for record in records:
        if isinstance(record, dict):
            role = str(record.get("role", "")).strip().lower()
            content = str(record.get("content", ""))
            raw_tool_calls = record.get("tool_calls_json", "[]")
        else:
            role = str(getattr(record, "role", "")).strip().lower()
            content = str(getattr(record, "content", ""))
            raw_tool_calls = getattr(record, "tool_calls_json", "[]")
        if role == "user":
            restored.append(HumanMessage(content=content))
            continue
        if role == "assistant":
            try:
                parsed = json.loads(raw_tool_calls)
            except Exception:
                parsed = []
            tool_calls = parsed if isinstance(parsed, list) else []
            restored.append(AIMessage(content=content, tool_calls=tool_calls))
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
    return restored


async def _apply_restored_messages(engine: Any, restored: list[Any]) -> None:
    if not restored:
        return
    await engine._agent.aupdate_state(
        {"configurable": {"thread_id": engine._thread_id}},
        {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *restored]},
        as_node="llm",
    )
