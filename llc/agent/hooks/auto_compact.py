from __future__ import annotations

from llc.agent.compact import compact_history
from llc.models import AvailableModel

from .base import HookContext
from .ordering import HOOK_ORDER_AUTO_COMPACT


class AutoCompactHook:
    order = HOOK_ORDER_AUTO_COMPACT

    async def after_turn(self, ctx: HookContext) -> str | None:
        if ctx.last_turn_input_tokens <= 0:
            return None

        context_length = _context_length_for_model(
            ctx.available_models,
            ctx.settings.model_name,
        )
        if context_length is None:
            return None

        threshold = int(context_length * 0.9)
        if ctx.last_turn_input_tokens <= threshold:
            return None

        compacted_count = await compact_history(
            ctx.agent,
            ctx.thread_id,
            ctx.settings,
        )
        if compacted_count <= 0:
            return None

        return f"Chat history auto-compacted ({compacted_count} messages summarized)."


def _context_length_for_model(
    models: list[AvailableModel],
    model_name: str,
) -> int | None:
    for model in models:
        if model.id == model_name:
            return model.context_length
    return None
