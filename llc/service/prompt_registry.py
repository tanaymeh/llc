from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class PromptRegistry:
    def __init__(self, prompts_dir: Path) -> None:
        self._prompts_dir = prompts_dir.resolve()
        self._cache: dict[tuple[str, str | None], str] = {}

    def get(self, name: str, *, key: str | None = None) -> str:
        cache_key = (name, key)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        path = self._resolve_path(name)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")

        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Prompt file must contain a mapping: {path}")

        prompt = self._extract_prompt(raw, name, key)
        prompt = prompt.strip()
        if not prompt:
            raise ValueError(f"Prompt is empty in {path}")

        self._cache[cache_key] = prompt
        return prompt

    def get_formatted(
        self,
        name: str,
        *,
        key: str | None = None,
        **kwargs: str,
    ) -> str:
        template = self.get(name, key=key)
        return template.format(**kwargs)

    def _resolve_path(self, name: str) -> Path:
        filename = name if name.endswith(".yaml") else f"{name}.yaml"
        return self._prompts_dir / filename

    @staticmethod
    def _extract_prompt(
        payload: dict[str, Any],
        name: str,
        key: str | None,
    ) -> str:
        if key:
            value = payload.get(key)
            if isinstance(value, str):
                return value
            raise ValueError(f"Missing or invalid prompt key '{key}' in '{name}'")

        key_candidates = (
            name,
            f"{name}_prompt",
            "prompt",
        )
        for candidate in key_candidates:
            value = payload.get(candidate)
            if isinstance(value, str):
                return value

        text_values = [value for value in payload.values() if isinstance(value, str)]
        if len(text_values) == 1:
            return text_values[0]
        raise ValueError(
            f"Cannot infer prompt key for '{name}'. Provide key explicitly."
        )

