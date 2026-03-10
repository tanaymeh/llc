from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from agent.compact import compact_history
from config import Settings
from models import AvailableModel, get_model_pricing

_DEBUG_LOG_PATH = Path("/home/tanay/Desktop/local-claude-code/.cursor/debug-eda7ca.log")
_DEBUG_LOG_FALLBACK_PATH = Path("/workspace/.cursor/debug-eda7ca.log")


# region debug token logs
def _debug_log(
    location: str,
    message: str,
    data: dict[str, Any],
    hypothesis_id: str,
    run_id: str,
) -> None:
    payload = {
        "sessionId": "eda7ca",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    for path in (_DEBUG_LOG_PATH, _DEBUG_LOG_FALLBACK_PATH):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, separators=(",", ":")) + "\n")
            break
        except Exception:
            continue


# endregion debug token logs


class Hook(Protocol):
    async def after_turn(self, ctx: "HookContext") -> str | None: ...


class HookContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent: Any
    thread_id: str
    settings: Settings
    last_turn_input_tokens: int
    last_turn_output_tokens: int
    available_models: list[AvailableModel]
    compact_prompt: str


class TokenCounterHook:
    def __init__(self) -> None:
        self.session_input = 0
        self.session_output = 0
        self.session_cost = 0.0

    async def after_turn(self, ctx: HookContext) -> str | None:
        before_input = self.session_input
        before_output = self.session_output
        self.session_input += ctx.last_turn_input_tokens
        self.session_output += ctx.last_turn_output_tokens
        if ctx.last_turn_input_tokens > 0 or ctx.last_turn_output_tokens > 0:
            pricing = get_model_pricing(
                ctx.available_models, ctx.settings.model_name
            )
            if pricing is not None:
                self.session_cost += (
                    ctx.last_turn_input_tokens * pricing[0]
                    + ctx.last_turn_output_tokens * pricing[1]
                )
        # region debug token logs
        _debug_log(
            "hooks.py:TokenCounterHook.after_turn",
            "token hook updated session totals",
            {
                "turn_input": ctx.last_turn_input_tokens,
                "turn_output": ctx.last_turn_output_tokens,
                "before_input": before_input,
                "before_output": before_output,
                "after_input": self.session_input,
                "after_output": self.session_output,
            },
            "H3",
            ctx.thread_id,
        )
        # endregion debug token logs
        return None


class AutoCompactHook:
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
