from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from llc.agent.message_utils import message_text
from llc.service.events import ReasoningDelta, TextDelta, ToolCallStarted, ToolResultEvent


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


def _extract_reasoning(message_chunk: Any) -> str:
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


def _token_usage(message: Any) -> tuple[int, int]:
    usage = getattr(message, "usage_metadata", None)
    if not usage:
        return 0, 0
    if isinstance(usage, dict):
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        return max(input_tokens, 0), max(output_tokens, 0)
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    return max(input_tokens, 0), max(output_tokens, 0)


class StreamAdapter:
    def __init__(self) -> None:
        self._seen_tool_call_ids: set[str] = set()
        self._seen_tool_result_ids: set[str] = set()
        self._tool_index = 0
        self._turn_input_tokens = 0
        self._turn_output_tokens = 0
        self._usage_updated = False
        self._fallback_text = ""
        self._streamed_text_chars = 0

    @property
    def turn_input_tokens(self) -> int:
        return self._turn_input_tokens

    @property
    def turn_output_tokens(self) -> int:
        return self._turn_output_tokens

    @property
    def streamed_text_chars(self) -> int:
        return self._streamed_text_chars

    @property
    def fallback_text(self) -> str:
        return self._fallback_text

    def consume_usage_updated(self) -> bool:
        updated = self._usage_updated
        self._usage_updated = False
        return updated

    def parse(self, mode: str, chunk: Any) -> list[Any]:
        events: list[Any] = []
        if mode == "messages":
            return self._parse_messages_chunk(chunk)
        if mode == "updates":
            return self._parse_updates_chunk(chunk)
        return events

    def _parse_messages_chunk(self, chunk: Any) -> list[Any]:
        if not isinstance(chunk, tuple) or len(chunk) != 2:
            return []
        message_chunk, meta = chunk
        if not isinstance(meta, dict) or meta.get("langgraph_node") != "llm":
            return []

        reasoning = _extract_reasoning(message_chunk)
        if reasoning:
            return [ReasoningDelta(text=reasoning)]

        text = message_text(getattr(message_chunk, "content", ""))
        if not text:
            return []
        self._streamed_text_chars += len(text)
        return [TextDelta(text=text)]

    def _parse_updates_chunk(self, chunk: Any) -> list[Any]:
        if not isinstance(chunk, dict):
            return []

        events: list[Any] = []
        for node_update in chunk.values():
            if not isinstance(node_update, dict):
                continue
            for message in node_update.get("messages", []):
                if isinstance(message, AIMessage):
                    usage_input, usage_output = _token_usage(message)
                    if usage_input or usage_output:
                        self._turn_input_tokens += usage_input
                        self._turn_output_tokens += usage_output
                        self._usage_updated = True

                    tool_calls = getattr(message, "tool_calls", None) or []
                    if not tool_calls:
                        text = message_text(message.content)
                        if text:
                            self._fallback_text = text

                    for tool_call in tool_calls:
                        tool_call_id = tool_call.get("id")
                        if not isinstance(tool_call_id, str) or not tool_call_id:
                            continue
                        if tool_call_id in self._seen_tool_call_ids:
                            continue
                        self._seen_tool_call_ids.add(tool_call_id)
                        self._tool_index += 1
                        raw_name = tool_call.get("name", "tool")
                        raw_args = tool_call.get("args", {})
                        tool_name = str(raw_name) if raw_name is not None else "tool"
                        args = raw_args if isinstance(raw_args, dict) else {}
                        events.append(
                            ToolCallStarted(
                                tool_call_id=tool_call_id,
                                tool_name=tool_name,
                                tool_index=self._tool_index,
                                args=args,
                            )
                        )
                    continue

                if not isinstance(message, ToolMessage):
                    continue
                tool_call_id = getattr(message, "tool_call_id", "")
                if not isinstance(tool_call_id, str) or not tool_call_id:
                    continue
                if tool_call_id in self._seen_tool_result_ids:
                    continue
                self._seen_tool_result_ids.add(tool_call_id)
                kwargs = getattr(message, "additional_kwargs", {}) or {}
                tool_name = str(kwargs.get("tool_name", "tool"))
                render_mode = str(kwargs.get("tool_render_mode", "") or "")
                user_facing = bool(kwargs.get("user_facing"))
                events.append(
                    ToolResultEvent(
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        content=message_text(message.content),
                        user_facing=user_facing,
                        render_mode=render_mode,
                    )
                )
        return events