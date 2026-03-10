from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from llc.agent.llm import build_chat_model
from llc.agent.message_utils import message_text
from llc.config import Settings

_MIN_COMPACT_MESSAGES = 6
_SUMMARY_PREFIX = "Summary of earlier conversation:\n\n"
_OVERSIZED_MESSAGE_CHARS = 12000


async def aget_history_messages(agent: Any, thread_id: str) -> list[AnyMessage]:
    state = await agent.aget_state(_graph_config(thread_id))
    values = getattr(state, "values", {})
    messages = values.get("messages", [])
    if not isinstance(messages, list):
        return []
    return list(messages)


async def compact_history(agent: Any, thread_id: str, settings: Settings) -> int:
    messages = await aget_history_messages(agent, thread_id)
    if len(messages) < _MIN_COMPACT_MESSAGES:
        return 0

    split_index = _find_split_index(messages)
    split_index = _adjust_split_for_oversized_messages(messages, split_index)
    if split_index <= 0 or split_index >= len(messages):
        return 0

    summary = await _summarize_messages(messages[:split_index], settings)
    retained = [_clone_message(message) for message in messages[split_index:]]
    replacement = [
        RemoveMessage(id=REMOVE_ALL_MESSAGES),
        SystemMessage(content=f"{_SUMMARY_PREFIX}{summary}"),
        *retained,
    ]
    await agent.aupdate_state(
        _graph_config(thread_id),
        {"messages": replacement},
        as_node="llm",
    )
    return split_index


def _graph_config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


def _find_split_index(messages: list[AnyMessage], fraction: float = 0.75) -> int:
    if len(messages) < _MIN_COMPACT_MESSAGES:
        return 0

    target = min(max(int(len(messages) * fraction), 1), len(messages) - 1)
    for idx in range(target, 0, -1):
        if _is_boundary_message(messages[idx]):
            return idx
    for idx in range(target + 1, len(messages)):
        if _is_boundary_message(messages[idx]):
            return idx
    return target


def _adjust_split_for_oversized_messages(
    messages: list[AnyMessage],
    split_index: int,
) -> int:
    if split_index <= 0 or split_index >= len(messages):
        return split_index

    # If a very large message is in the retained tail, include it in the
    # compacted slice when possible so compaction can actually reduce context.
    for idx in range(split_index, len(messages) - 1):
        content = message_text(messages[idx].content, include_reasoning=True).strip()
        if len(content) <= _OVERSIZED_MESSAGE_CHARS:
            continue
        return max(split_index, min(idx + 1, len(messages) - 1))
    return split_index


def _is_boundary_message(message: AnyMessage) -> bool:
    return isinstance(message, HumanMessage | SystemMessage)


def _clone_message(message: AnyMessage) -> AnyMessage:
    return message.model_copy(update={"id": None})


async def _summarize_messages(messages: list[AnyMessage], settings: Settings) -> str:
    model = build_chat_model(settings, settings.compact_model_name)
    response = await model.ainvoke(
        [
            SystemMessage(content=settings.compact_prompt),
            HumanMessage(content=_render_messages(messages)),
        ]
    )
    summary = message_text(response.content, include_reasoning=True).strip()
    if not summary:
        raise ValueError("Compaction summary model returned empty content")
    return summary


def _render_messages(messages: list[AnyMessage]) -> str:
    lines: list[str] = []
    for idx, message in enumerate(messages, start=1):
        lines.append(f"[{idx}] {_message_heading(message)}")
        content = message_text(message.content, include_reasoning=True)
        if content:
            lines.append(content)
        tool_details = _tool_details(message)
        if tool_details:
            lines.append(tool_details)
        lines.append("")
    return "\n".join(lines).strip()


def _message_heading(message: AnyMessage) -> str:
    if isinstance(message, HumanMessage):
        return "User"
    if isinstance(message, ToolMessage):
        name = getattr(message, "name", None) or "tool"
        return f"Tool: {name}"
    if isinstance(message, SystemMessage):
        return "System"
    return "Assistant"


def _tool_details(message: AnyMessage) -> str:
    if isinstance(message, AIMessage):
        tool_calls = getattr(message, "tool_calls", None) or []
        if not tool_calls:
            return ""
        rendered: list[str] = []
        for tool_call in tool_calls:
            name = tool_call.get("name", "tool")
            args = tool_call.get("args", {})
            args_json = json.dumps(args, sort_keys=True, default=str)
            rendered.append(f"Tool call: {name}({args_json})")
        return "\n".join(rendered)

    if isinstance(message, ToolMessage):
        tool_call_id = getattr(message, "tool_call_id", None)
        if tool_call_id:
            return f"Tool call id: {tool_call_id}"
    return ""


