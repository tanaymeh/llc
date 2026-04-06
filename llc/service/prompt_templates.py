from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound

PROMPT_MANIFEST_FILENAME = "prompt_manifest.yaml"


@dataclass(frozen=True)
class PromptTemplateSpec:
    file: str
    variables: tuple[str, ...]


class PromptTemplates:
    def __init__(self, prompts_dir: Path) -> None:
        self._prompts_dir = prompts_dir.resolve()
        self._manifest_path = self._prompts_dir / PROMPT_MANIFEST_FILENAME
        self._specs = self._load_manifest(self._manifest_path)
        self._environment = Environment(
            loader=FileSystemLoader(str(self._prompts_dir), encoding="utf-8"),
            undefined=StrictUndefined,
            autoescape=False,
            keep_trailing_newline=True,
        )

    def render(
        self,
        name: str,
        *,
        key: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> str:
        prompt_id, spec = self._resolve(name, key)
        payload = dict(context or {})
        missing = [variable for variable in spec.variables if variable not in payload]
        if missing:
            missing_names = ", ".join(missing)
            raise ValueError(
                f"Missing prompt variables for '{prompt_id}': {missing_names}"
            )

        unexpected = sorted(set(payload) - set(spec.variables))
        if unexpected:
            unexpected_names = ", ".join(unexpected)
            raise ValueError(
                f"Unexpected prompt variables for '{prompt_id}': {unexpected_names}"
            )

        try:
            template = self._environment.get_template(spec.file)
        except TemplateNotFound as exc:
            raise FileNotFoundError(
                f"Prompt template not found for '{prompt_id}': {spec.file}"
            ) from exc

        rendered = template.render(**payload).strip()
        if not rendered:
            raise ValueError(f"Prompt is empty for '{prompt_id}'")
        return rendered

    def expected_variables(
        self,
        name: str,
        *,
        key: str | None = None,
    ) -> tuple[str, ...]:
        _, spec = self._resolve(name, key)
        return spec.variables

    def _resolve(self, name: str, key: str | None) -> tuple[str, PromptTemplateSpec]:
        prompt_id = self._prompt_id(name, key)
        spec = self._specs.get(prompt_id)
        if spec is not None:
            return prompt_id, spec

        if key is not None:
            raise FileNotFoundError(f"Prompt definition not found: {prompt_id}")

        prefix = f"{name}:"
        matches = [
            candidate_id
            for candidate_id in self._specs
            if candidate_id == name or candidate_id.startswith(prefix)
        ]
        if len(matches) == 1:
            match = matches[0]
            return match, self._specs[match]
        raise FileNotFoundError(f"Prompt definition not found: {name}")

    @staticmethod
    def _prompt_id(name: str, key: str | None) -> str:
        clean_name = name.strip()
        clean_key = key.strip() if key else ""
        return clean_name if not clean_key else f"{clean_name}:{clean_key}"

    def _load_manifest(self, path: Path) -> dict[str, PromptTemplateSpec]:
        if not path.exists():
            raise FileNotFoundError(f"Prompt manifest not found: {path}")

        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Prompt manifest must contain a mapping: {path}")

        prompts = raw.get("prompts")
        if not isinstance(prompts, dict) or not prompts:
            raise ValueError(
                f"Prompt manifest must define a non-empty 'prompts' mapping: {path}"
            )

        variable_map: dict[str, list[str]] = {}
        raw_variables = raw.get("variables", {})
        if raw_variables not in ({}, None) and not isinstance(raw_variables, dict):
            raise ValueError(
                f"Prompt manifest 'variables' must be a mapping: {path}"
            )

        specs: dict[str, PromptTemplateSpec] = {}
        for prompt_id, payload in prompts.items():
            if not isinstance(prompt_id, str) or not prompt_id.strip():
                raise ValueError(
                    f"Prompt manifest contains an invalid prompt id: {path}"
                )
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Prompt definition for '{prompt_id}' must be a mapping: {path}"
                )
            file_name = payload.get("file")
            if not isinstance(file_name, str) or not file_name.strip():
                raise ValueError(
                    f"Prompt definition for '{prompt_id}' must define "
                    f"a non-empty file: {path}"
                )
            normalized_file = self._normalize_file(file_name.strip())
            specs[prompt_id] = PromptTemplateSpec(file=normalized_file, variables=())
            variable_map[prompt_id] = []

        if isinstance(raw_variables, dict):
            for variable_name, prompt_ids in raw_variables.items():
                if not isinstance(variable_name, str) or not variable_name.strip():
                    raise ValueError(
                        f"Prompt manifest contains an invalid variable name: {path}"
                    )
                if not isinstance(prompt_ids, list):
                    raise ValueError(
                        f"Variable '{variable_name}' must map to a list of prompts: {path}"
                    )
                for prompt_id in prompt_ids:
                    if not isinstance(prompt_id, str) or prompt_id not in specs:
                        raise ValueError(
                            f"Variable '{variable_name}' references unknown prompt '{prompt_id}'"
                        )
                    variable_map[prompt_id].append(variable_name)

        return {
            prompt_id: PromptTemplateSpec(
                file=spec.file,
                variables=tuple(variable_map[prompt_id]),
            )
            for prompt_id, spec in specs.items()
        }

    def _normalize_file(self, file_name: str) -> str:
        candidate = (self._prompts_dir / file_name).resolve()
        candidate.relative_to(self._prompts_dir)
        if not candidate.exists():
            raise FileNotFoundError(f"Prompt template file not found: {candidate}")
        if candidate.suffix != ".jinja":
            raise ValueError(
                f"Prompt template files must use the .jinja extension: {candidate}"
            )
        return candidate.relative_to(self._prompts_dir).as_posix()
