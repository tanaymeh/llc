from __future__ import annotations

from .base import Hook

HOOK_ORDER_TOKEN_COUNTER = 10
HOOK_ORDER_TOOL_OUTPUT_SUMMARY = 20
HOOK_ORDER_AUTO_COMPACT = 30
DEFAULT_HOOK_ORDER = 100


def hook_order(hook: Hook) -> int:
    raw = getattr(hook, "order", DEFAULT_HOOK_ORDER)
    try:
        return int(raw)
    except Exception:
        return DEFAULT_HOOK_ORDER
