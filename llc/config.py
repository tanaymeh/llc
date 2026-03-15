import os
import platform
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field
import yaml

SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "system_prompt.yaml"
COMPACT_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "compact_prompt.yaml"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


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


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _env_path(name: str) -> Path | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    return Path(raw).expanduser().resolve()


class Settings(BaseModel, frozen=True):
    model_name: str = "gpt-4o-mini"
    compact_model_name: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    firecrawl_api_key: str | None = None
    workspace_root: Path = Field(default_factory=lambda: Path.cwd().resolve())
    prompts_dir: Path = PROMPTS_DIR
    db_path: Path = Field(
        default_factory=lambda: (Path.cwd().resolve() / ".llc" / "sessions.db")
    )
    system_prompt: str = Field(default_factory=_load_system_prompt)
    compact_prompt: str = Field(default_factory=_load_compact_prompt)
    sub_agent_mode_enabled: bool = False
    max_sub_agents: int = Field(default=5, ge=1, le=5)
    sub_agent_report_interval_s: int = Field(default=8, ge=1, le=300)
    sub_agent_max_runtime_s: int = Field(default=900, ge=30, le=7200)
    sub_agent_context_messages: int = Field(default=8, ge=1, le=20)
    shell_timeout: int = 120

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        workspace_root = Path.cwd().resolve()
        prompts_dir = _env_path("LLC_PROMPTS_DIR") or PROMPTS_DIR
        system_prompt = _load_prompt(prompts_dir / "system_prompt.yaml", "system_prompt")
        compact_prompt = _load_prompt(prompts_dir / "compact_prompt.yaml", "compact_prompt")
        db_path = _env_path("LLC_DB_PATH") or (workspace_root / ".llc" / "sessions.db")
        return cls(
            model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
            compact_model_name=os.getenv("COMPACT_MODEL_NAME"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
            firecrawl_api_key=os.getenv("FIRECRAWL_API_KEY"),
            workspace_root=workspace_root,
            prompts_dir=prompts_dir,
            db_path=db_path,
            system_prompt=_with_env_details(
                system_prompt,
                workspace_root,
            ),
            compact_prompt=compact_prompt,
            sub_agent_mode_enabled=_env_bool("SUB_AGENT_MODE_ENABLED", default=False),
            max_sub_agents=_env_int("MAX_SUB_AGENTS", 5),
            sub_agent_report_interval_s=_env_int("SUB_AGENT_REPORT_INTERVAL_S", 8),
            sub_agent_max_runtime_s=_env_int("SUB_AGENT_MAX_RUNTIME_S", 900),
            sub_agent_context_messages=_env_int("SUB_AGENT_CONTEXT_MESSAGES", 8),
        )
