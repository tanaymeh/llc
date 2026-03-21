from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from llc.config import Settings
from llc.models import AvailableModel

HookExecutionMode = Literal["background", "blocking"]


class Hook(Protocol):
    name: str
    order: int
    execution_mode: HookExecutionMode

    async def prepare_after_turn(self, ctx: "HookContext") -> Any | None: ...

    async def apply_prepared(self, ctx: "HookContext", prepared: Any) -> str | None: ...


class HookContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent: Any
    thread_id: str
    settings: Settings
    last_turn_input_tokens: int
    last_turn_output_tokens: int
    available_models: list[AvailableModel]
    compact_prompt: str
    tool_output_candidates: dict[str, tuple[str, str]] = Field(default_factory=dict)
