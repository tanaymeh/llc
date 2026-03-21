from __future__ import annotations

from .base import HookContext
from .ordering import HOOK_ORDER_TOKEN_COUNTER
from llc.models import get_model_pricing


class TokenCounterHook:
    order = HOOK_ORDER_TOKEN_COUNTER

    def __init__(self) -> None:
        self.session_input = 0
        self.session_output = 0
        self.session_cost = 0.0

    async def after_turn(self, ctx: HookContext) -> str | None:
        self.session_input += ctx.last_turn_input_tokens
        self.session_output += ctx.last_turn_output_tokens
        if ctx.last_turn_input_tokens > 0 or ctx.last_turn_output_tokens > 0:
            pricing = get_model_pricing(
                ctx.available_models,
                ctx.settings.model_name,
            )
            if pricing is not None:
                self.session_cost += (
                    ctx.last_turn_input_tokens * pricing[0]
                    + ctx.last_turn_output_tokens * pricing[1]
                )
        return None
