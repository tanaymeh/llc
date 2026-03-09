import os
import platform
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field
import yaml

SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "system_prompt.yaml"
COMPACT_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "compact_prompt.yaml"


def _load_prompt(path: Path, key: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt required at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    prompt = data.get(key)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(
            f"{path.relative_to(Path(__file__).resolve().parent)} must contain "
            f"a non-empty '{key}' key"
        )
    return prompt.strip()


def _load_system_prompt(path: Path = SYSTEM_PROMPT_PATH) -> str:
    return _load_prompt(path, "system_prompt")


def _load_compact_prompt(path: Path = COMPACT_PROMPT_PATH) -> str:
    return _load_prompt(path, "compact_prompt")


def _find_git_root(path: Path) -> Path | None:
    for current in (path, *path.parents):
        if (current / ".git").exists():
            return current
    return None


def _format_today(now: datetime) -> str:
    return f"{now.strftime('%A %b')} {now.day}, {now.year}"


def _build_env_details(workspace_root: Path) -> str:
    git_root = _find_git_root(workspace_root)
    git_status = f"Yes, at {git_root}" if git_root else "No"
    shell = os.getenv("SHELL") or "unknown"
    os_version = f"{platform.system().lower()} {platform.release()}"
    today = _format_today(datetime.now().astimezone())
    lines = [
        f"OS Version: {os_version}",
        f"Shell: {shell}",
        f"Workspace Path: {workspace_root}",
        f"Is directory a git repo: {git_status}",
        f"Today's date: {today}",
    ]
    return "\n".join(lines)


def _with_env_details(system_prompt: str, workspace_root: Path) -> str:
    env_details = _build_env_details(workspace_root)
    return f"{system_prompt.rstrip()}\n\n<env>\n{env_details}\n</env>"


class Settings(BaseModel, frozen=True):
    model_name: str = "gpt-4o-mini"
    compact_model_name: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    firecrawl_api_key: str | None = None
    workspace_root: Path = Field(default_factory=lambda: Path.cwd().resolve())
    system_prompt: str = Field(default_factory=_load_system_prompt)
    compact_prompt: str = Field(default_factory=_load_compact_prompt)
    shell_timeout: int = 120

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        workspace_root = Path.cwd().resolve()
        return cls(
            model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
            compact_model_name=os.getenv("COMPACT_MODEL_NAME"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
            firecrawl_api_key=os.getenv("FIRECRAWL_API_KEY"),
            workspace_root=workspace_root,
            system_prompt=_with_env_details(
                _load_system_prompt(),
                workspace_root,
            ),
        )
