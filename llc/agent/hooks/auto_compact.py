from __future__ import annotations

from llc.agent.compact import CompactionPlan, apply_compaction_plan, build_compaction_plan
from llc.models import AvailableModel

from .base import HookContext
from .ordering import HOOK_ORDER_AUTO_COMPACT


class AutoCompactHook:
    name = "auto_compact"
    order = HOOK_ORDER_AUTO_COMPACT
    execution_mode = "background"

    async def prepare_after_turn(self, ctx: HookContext) -> CompactionPlan | None:
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

        return await build_compaction_plan(
            ctx.agent,
            ctx.thread_id,
            ctx.settings,
        )

    async def apply_prepared(
        self,
        ctx: HookContext,
        prepared: CompactionPlan,
    ) -> str | None:
        compacted_count = await apply_compaction_plan(
            ctx.agent,
            ctx.thread_id,
            prepared,
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
