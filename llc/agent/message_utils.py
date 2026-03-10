from __future__ import annotations

from typing import Any


def message_text(content: Any, *, include_reasoning: bool = False) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            if include_reasoning:
                parts.append(str(block))
            continue
        block_type = str(block.get("type", "")).lower()
        if block_type in {"thinking", "reasoning"} and not include_reasoning:
            continue
        raw = (
            block.get("text")
            or block.get("content")
            or block.get("output_text")
            or block.get("reasoning")
            or block.get("thinking")
        )
        if raw:
            parts.append(str(raw))
    return "\n".join(part for part in parts if part)
