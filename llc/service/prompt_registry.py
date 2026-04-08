from __future__ import annotations

from pathlib import Path
from typing import Any

from llc.service.prompt_templates import PromptTemplates


class PromptRegistry:
    def __init__(self, prompts_dir: Path) -> None:
        self._prompts_dir = prompts_dir.resolve()
        self._cache: dict[tuple[str, str | None], str] = {}
        self._templates = PromptTemplates(self._prompts_dir)

    def get(self, name: str, *, key: str | None = None) -> str:
        cache_key = (name, key)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        prompt = self._templates.render(name, key=key)
        self._cache[cache_key] = prompt
        return prompt

    def get_formatted(
        self,
        name: str,
        *,
        key: str | None = None,
        **kwargs: Any,
    ) -> str:
        return self._templates.render(name, key=key, context=kwargs)

