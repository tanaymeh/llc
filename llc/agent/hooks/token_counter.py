from __future__ import annotations

from .base import HookContext
from .ordering import HOOK_ORDER_TOKEN_COUNTER
from llc.models import get_model_pricing


class TokenCounterHook:
    name = "token_counter"
    order = HOOK_ORDER_TOKEN_COUNTER
    execution_mode = "blocking"

    def __init__(self) -> None:
        self.session_input = 0
        self.session_output = 0
        self.session_cost = 0.0

    async def prepare_after_turn(self, ctx: HookContext) -> tuple[int, int, float]:
        incremental_cost = 0.0
        if ctx.last_turn_input_tokens > 0 or ctx.last_turn_output_tokens > 0:
            pricing = get_model_pricing(
                ctx.available_models,
                ctx.settings.model_name,
            )
            if pricing is not None:
                incremental_cost = (
                    ctx.last_turn_input_tokens * pricing[0]
                    + ctx.last_turn_output_tokens * pricing[1]
                )
        return (
            ctx.last_turn_input_tokens,
            ctx.last_turn_output_tokens,
            incremental_cost,
        )

    async def apply_prepared(
        self,
        ctx: HookContext,
        prepared: tuple[int, int, float],
    ) -> str | None:
        del ctx
        input_tokens, output_tokens, incremental_cost = prepared
        self.session_input += input_tokens
        self.session_output += output_tokens
        self.session_cost += incremental_cost
        return None
