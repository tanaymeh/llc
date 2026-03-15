from __future__ import annotations

from typing import Any


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

