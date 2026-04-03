from langchain_core.tools import BaseTool

from llc.agent.tools.code_grep import make_code_grep_tools
from llc.agent.tools.edit import make_edit_tools
from llc.agent.tools.filesystem import make_filesystem_tools
from llc.agent.tools.glob import make_glob_tools
from llc.agent.tools.grep import make_grep_tools
from llc.agent.tools.shell import make_shell_tools
from llc.agent.tools.subagent_coordination import make_subagent_coordination_tools
from llc.agent.tools.subagents import make_subagent_tools
from llc.agent.tools.todo import make_todo_tools
from llc.agent.tools.web import make_web_tools
from llc.config import Settings


_SUBAGENT_SHELL_TIMEOUT_BUFFER_S = 5


def _effective_shell_timeout(settings: Settings, *, role: str) -> int:
    timeout = max(int(settings.shell_timeout), 1)
    if role != "subagent":
        return timeout
    stall_timeout = max(int(settings.sub_agent_stall_timeout_s), 1)
    max_subagent_timeout = max(stall_timeout - _SUBAGENT_SHELL_TIMEOUT_BUFFER_S, 1)
    return min(timeout, max_subagent_timeout)


def collect_tools(
    settings: Settings,
    *,
    role: str = "default",
    subagent_runtime: object | None = None,
    subagent_coordination: object | None = None,
    subagent_id: str = "",
) -> list[BaseTool]:
    mutation_guard = None
    if role == "subagent" and subagent_coordination is not None and subagent_id.strip():
        ensure_mutation_allowed = getattr(
            subagent_coordination,
            "ensure_mutation_allowed",
            None,
        )
        if callable(ensure_mutation_allowed):
            clean_subagent_id = subagent_id.strip()

            def _guard(path: str) -> str | None:
                try:
                    verdict = ensure_mutation_allowed(
                        agent_id=clean_subagent_id,
                        file_path=path,
                    )
                except Exception as exc:  # noqa: BLE001
                    return f"mutation lock check failed: {exc!s}"
                if not isinstance(verdict, dict):
                    return "mutation lock check failed."
                if bool(verdict.get("ok", False)):
                    return None
                return str(verdict.get("error", "mutation lock required."))

            mutation_guard = _guard

    tools: list[BaseTool] = [
        *make_filesystem_tools(
            settings.workspace_root,
            mutation_guard=mutation_guard,
        ),
        *make_shell_tools(_effective_shell_timeout(settings, role=role)),
        *make_glob_tools(settings.workspace_root),
        *make_grep_tools(settings.workspace_root),
        *make_edit_tools(
            settings.workspace_root,
            mutation_guard=mutation_guard,
        ),
        *make_todo_tools(),
        *make_web_tools(settings.firecrawl_api_key),
        *make_code_grep_tools(
            settings.workspace_root,
            mutation_guard=mutation_guard,
        ),
    ]
    if (
        role == "orchestrator"
        and settings.sub_agent_mode_enabled
        and subagent_runtime is not None
    ):
        tools.extend(make_subagent_tools(subagent_runtime))
    if (
        role == "subagent"
        and settings.sub_agent_mode_enabled
        and subagent_coordination is not None
        and subagent_id.strip()
    ):
        tools.extend(
            make_subagent_coordination_tools(
                subagent_coordination,
                subagent_id=subagent_id,
                workspace_root=settings.workspace_root,
            )
        )
    return tools
