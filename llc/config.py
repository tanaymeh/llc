import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from llc.service.prompt_templates import PromptTemplates

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_PROMPT_TEMPLATES: dict[Path, PromptTemplates] = {}
_TREE_MAX_DEPTH = 3
_TREE_MAX_ENTRIES = 200
_GIT_LOG_MAX_COMMITS = 10
_TREE_IGNORED_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".next",
    "dist",
    "build",
}


def _prompt_templates(prompts_dir: Path) -> PromptTemplates:
    resolved = prompts_dir.resolve()
    cached = _PROMPT_TEMPLATES.get(resolved)
    if cached is not None:
        return cached
    templates = PromptTemplates(resolved)
    _PROMPT_TEMPLATES[resolved] = templates
    return templates


def _render_prompt(
    prompts_dir: Path,
    name: str,
    *,
    context: Mapping[str, Any] | None = None,
) -> str:
    return _prompt_templates(prompts_dir).render(name, context=context)


def _load_system_prompt(
    prompts_dir: Path = PROMPTS_DIR,
    *,
    env_block: str = "",
) -> str:
    return _render_prompt(
        prompts_dir,
        "system_prompt",
        context={"env_block": env_block},
    )


def _load_compact_prompt(prompts_dir: Path = PROMPTS_DIR) -> str:
    return _render_prompt(prompts_dir, "compact_prompt")


def _find_git_root(path: Path) -> Path | None:
    for current in (path, *path.parents):
        if (current / ".git").exists():
            return current
    return None


def _format_today(now: datetime) -> str:
    return f"{now.strftime('%A %b')} {now.day}, {now.year}"


def _run_git(
    workspace_root: Path,
    *args: str,
) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=workspace_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.SubprocessError,
        OSError,
    ):
        return None
    output = result.stdout.strip()
    return output or None


def _is_ignored_tree_entry(path: Path) -> bool:
    return path.name in _TREE_IGNORED_NAMES


def _build_directory_tree(root: Path) -> str:
    lines = [f"{root.name}/"]
    entry_count = 1
    truncated = False

    def walk(directory: Path, prefix: str, depth: int) -> None:
        nonlocal entry_count, truncated
        if truncated or depth >= _TREE_MAX_DEPTH:
            return
        try:
            children = sorted(
                (
                    child
                    for child in directory.iterdir()
                    if not _is_ignored_tree_entry(child)
                ),
                key=lambda child: (not child.is_dir(), child.name.lower()),
            )
        except OSError:
            lines.append(f"{prefix}[unreadable]")
            return

        total = len(children)
        for index, child in enumerate(children):
            if entry_count >= _TREE_MAX_ENTRIES:
                truncated = True
                lines.append(f"{prefix}... [tree truncated]")
                return
            connector = "└── " if index == total - 1 else "├── "
            suffix = "/" if child.is_dir() else ""
            lines.append(f"{prefix}{connector}{child.name}{suffix}")
            entry_count += 1
            if child.is_dir():
                extension = "    " if index == total - 1 else "│   "
                walk(child, prefix + extension, depth + 1)
                if truncated:
                    return

    walk(root, "", 0)
    return "\n".join(lines)



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
        f"Current directory name: {workspace_root.name}",
        f"Is directory a git repo: {git_status}",
        f"Today's date: {today}",
    ]
    lines.extend(
        [
            "Current directory tree:",
            _build_directory_tree(workspace_root),
        ]
    )

    if git_root is not None:
        git_log = _run_git(
            workspace_root,
            "log",
            "--all",
            f"-n{_GIT_LOG_MAX_COMMITS}",
            "--date=short",
            "--pretty=format:%h %ad %d %s",
        )
        if git_log:
            lines.extend(
                [
                    "Last 10 commits across local and remote refs:",
                    git_log,
                ]
            )
    return "\n".join(lines)


def _build_env_block(workspace_root: Path) -> str:
    env_details = _build_env_details(workspace_root)
    return f"<env>\n{env_details}\n</env>"


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


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def _env_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    return values or default


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
    sub_agent_stall_timeout_s: int = Field(default=45, ge=5, le=3600)
    sub_agent_stop_grace_s: int = Field(default=15, ge=1, le=300)
    sub_agent_max_tool_calls: int = Field(default=80, ge=1, le=2000)
    sub_agent_require_tool_call: bool = False
    sub_agent_wait_timeout_ms: int = Field(default=1200, ge=100, le=60000)
    sub_agent_context_messages: int = Field(default=8, ge=1, le=20)
    sub_agent_team_status_interval_cycles: int = Field(default=4, ge=1, le=500)
    sub_agent_shared_notes_interval_cycles: int = Field(default=6, ge=1, le=500)
    sub_agent_lock_default_lease_s: int = Field(default=120, ge=5, le=3600)
    sub_agent_lock_renew_s: int = Field(default=30, ge=5, le=1200)
    sub_agent_lock_near_expiry_s: int = Field(default=20, ge=1, le=600)
    sub_agent_shared_notes_max_entries: int = Field(default=40, ge=1, le=500)
    sub_agent_inbox_read_max: int = Field(default=8, ge=1, le=100)
    sub_agent_debug_logging: bool = False
    shell_timeout: int = 120
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_allowed_origins: tuple[str, ...] = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    )
    api_subagent_report_interval_s: float = Field(default=1.0, ge=0.2, le=30.0)

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        workspace_root = Path.cwd().resolve()
        prompts_dir = _env_path("LLC_PROMPTS_DIR") or PROMPTS_DIR
        system_prompt = _load_system_prompt(
            prompts_dir,
            env_block=_build_env_block(workspace_root),
        )
        compact_prompt = _load_compact_prompt(prompts_dir)
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
            system_prompt=system_prompt,
            compact_prompt=compact_prompt,
            sub_agent_mode_enabled=_env_bool("SUB_AGENT_MODE_ENABLED", default=False),
            max_sub_agents=_env_int("MAX_SUB_AGENTS", 5),
            sub_agent_report_interval_s=_env_int("SUB_AGENT_REPORT_INTERVAL_S", 8),
            sub_agent_max_runtime_s=_env_int("SUB_AGENT_MAX_RUNTIME_S", 900),
            sub_agent_stall_timeout_s=_env_int("SUB_AGENT_STALL_TIMEOUT_S", 45),
            sub_agent_stop_grace_s=_env_int("SUB_AGENT_STOP_GRACE_S", 15),
            sub_agent_max_tool_calls=_env_int("SUB_AGENT_MAX_TOOL_CALLS", 80),
            sub_agent_require_tool_call=_env_bool("SUB_AGENT_REQUIRE_TOOL_CALL", default=False),
            sub_agent_wait_timeout_ms=_env_int("SUB_AGENT_WAIT_TIMEOUT_MS", 1200),
            sub_agent_context_messages=_env_int("SUB_AGENT_CONTEXT_MESSAGES", 8),
            sub_agent_team_status_interval_cycles=_env_int(
                "SUB_AGENT_TEAM_STATUS_INTERVAL_CYCLES",
                4,
            ),
            sub_agent_shared_notes_interval_cycles=_env_int(
                "SUB_AGENT_SHARED_NOTES_INTERVAL_CYCLES",
                6,
            ),
            sub_agent_lock_default_lease_s=_env_int(
                "SUB_AGENT_LOCK_DEFAULT_LEASE_S",
                120,
            ),
            sub_agent_lock_renew_s=_env_int("SUB_AGENT_LOCK_RENEW_S", 30),
            sub_agent_lock_near_expiry_s=_env_int("SUB_AGENT_LOCK_NEAR_EXPIRY_S", 20),
            sub_agent_shared_notes_max_entries=_env_int(
                "SUB_AGENT_SHARED_NOTES_MAX_ENTRIES",
                40,
            ),
            sub_agent_inbox_read_max=_env_int("SUB_AGENT_INBOX_READ_MAX", 8),
            sub_agent_debug_logging=_env_bool("SUB_AGENT_DEBUG_LOGGING", default=False),
            api_host=os.getenv("LLC_API_HOST", "127.0.0.1"),
            api_port=_env_int("LLC_API_PORT", 8000),
            api_allowed_origins=_env_csv(
                "LLC_API_ALLOWED_ORIGINS",
                ("http://localhost:5173", "http://127.0.0.1:5173"),
            ),
            api_subagent_report_interval_s=_env_float(
                "LLC_API_SUBAGENT_REPORT_INTERVAL_S",
                1.0,
            ),
        )
