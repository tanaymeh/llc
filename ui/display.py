from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, ToolMessage


def _stringify_reasoning(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            for key in (
                "text",
                "content",
                "reasoning",
                "reasoning_content",
                "summary",
            ):
                raw = item.get(key)
                if raw:
                    parts.append(str(raw))
                    break
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        for key in ("text", "content", "reasoning", "reasoning_content", "summary"):
            raw = value.get(key)
            if raw:
                return str(raw)
        return ""
    return str(value)


def extract_reasoning(message_chunk: Any) -> str:
    additional_kwargs = getattr(message_chunk, "additional_kwargs", {})
    if isinstance(additional_kwargs, dict):
        for key in (
            "reasoning_content",
            "reasoning_details",
            "reasoning_summary_chunk",
            "reasoning_summary",
            "reasoning",
        ):
            reasoning_text = _stringify_reasoning(additional_kwargs.get(key))
            if reasoning_text:
                return reasoning_text

    content_blocks = getattr(message_chunk, "content_blocks", None)
    if isinstance(content_blocks, list):
        parts: list[str] = []
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type", "")).lower()
            if block_type not in {"thinking", "reasoning"}:
                continue
            raw = (
                block.get("reasoning")
                or block.get("thinking")
                or block.get("reasoning_content")
                or block.get("text")
                or block.get("content")
            )
            reasoning_text = _stringify_reasoning(raw)
            if reasoning_text:
                parts.append(reasoning_text)
        if parts:
            return "\n".join(parts)

    content = getattr(message_chunk, "content", None)
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type", "")).lower()
        if block_type not in {"thinking", "reasoning"}:
            continue
        raw = (
            block.get("thinking")
            or block.get("reasoning_content")
            or block.get("reasoning")
            or block.get("text")
            or block.get("content")
        )
        reasoning_text = _stringify_reasoning(raw)
        if reasoning_text:
            parts.append(reasoning_text)
    return "\n".join(part for part in parts if part)


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    text_parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            block_type = str(block.get("type", "")).lower()
            if block_type in {"thinking", "reasoning"}:
                continue
            raw = block.get("text") or block.get("content") or block.get("output_text")
            if raw:
                text_parts.append(str(raw))
            continue
        if isinstance(block, str):
            text_parts.append(block)
    return "\n".join(part for part in text_parts if part)


def format_tool_args(args: dict[str, Any]) -> str:
    if not args:
        return ""
    parts: list[str] = []
    for key, value in args.items():
        val = repr(value) if isinstance(value, str) else str(value)
        if len(val) > 60:
            val = val[:57] + "..."
        parts.append(f"{key}={val}")
    return ", ".join(parts)


def collect_new_tool_calls(
    chunk: Any,
    seen_tool_call_ids: set[str],
) -> list[dict[str, Any]]:
    if not isinstance(chunk, dict):
        return []
    pending: list[dict[str, Any]] = []
    for node_update in chunk.values():
        if not isinstance(node_update, dict):
            continue
        for message in node_update.get("messages", []):
            if not isinstance(message, AIMessage) or not message.tool_calls:
                continue
            for tool_call in message.tool_calls:
                tool_id = tool_call.get("id")
                if not isinstance(tool_id, str):
                    continue
                if tool_id in seen_tool_call_ids:
                    continue
                seen_tool_call_ids.add(tool_id)
                pending.append(tool_call)
    return pending


def collect_new_user_facing_tool_results(
    chunk: Any,
    seen_tool_result_ids: set[str],
) -> list[dict[str, str]]:
    if not isinstance(chunk, dict):
        return []
    pending: list[dict[str, str]] = []
    for node_update in chunk.values():
        if not isinstance(node_update, dict):
            continue
        for message in node_update.get("messages", []):
            if not isinstance(message, ToolMessage):
                continue
            tool_call_id = getattr(message, "tool_call_id", "")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                continue
            if tool_call_id in seen_tool_result_ids:
                continue
            seen_tool_result_ids.add(tool_call_id)
            kwargs = getattr(message, "additional_kwargs", {}) or {}
            if not bool(kwargs.get("user_facing")):
                continue
            pending.append(
                {
                    "tool_name": str(kwargs.get("tool_name", "tool")),
                    "render_mode": str(kwargs.get("tool_render_mode", "") or ""),
                    "content": message_text(message.content),
                }
            )
    return pending
