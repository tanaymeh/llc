from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from llc.agent.llm import build_chat_model
from llc.agent.nodes import make_llm_node, make_tool_node, should_continue
from llc.agent.subagents.coordination import SubAgentCoordinationClient
from llc.agent.state import AgentState
from llc.agent.tools import collect_tools
from llc.config import Settings

if TYPE_CHECKING:
    from llc.service.prompt_registry import PromptRegistry

AgentRole = Literal["default", "orchestrator", "subagent"]
_MAX_SNAPSHOT_WORKERS = 5

_ORCHESTRATOR_MODE_APPEND = (
    "Mode: orchestrator.\n"
    "You may complete work yourself.\n"
    "Prefer delegation only when work is complex and cleanly divisible.\n"
    "Use running worker reports before intervening.\n"
    "If active workers exist, keep this response open until they finish.\n"
    "Do not ask the user to poll for worker updates.\n"
    "Always use the live worker snapshot in the system prompt for worker status.\n"
    "When calling LaunchSubagent, always set a short, descriptive `name`.\n"
    "Interrupt or terminate only when work is clearly off-track, unsafe, or stuck.\n"
    "CRITICAL — parallel launch: When you need multiple sub-agents, call LaunchSubagent "
    "for ALL of them in a SINGLE response (multiple tool calls in one turn). "
    "Do NOT launch one, wait, then launch the next. Sub-agents coordinate directly "
    "via peer messaging, shared notes, and scope claims."
)


def _prompt_for_role(
    settings: Settings,
    role: AgentRole,
    prompt_registry: PromptRegistry | None = None,
) -> str:
    base_prompt = settings.system_prompt.rstrip()
    if role == "orchestrator" and settings.sub_agent_mode_enabled:
        append = _ORCHESTRATOR_MODE_APPEND
        if prompt_registry is not None:
            try:
                append = prompt_registry.get(
                    "orchestrator_mode",
                    key="orchestrator_mode_prompt",
                )
            except Exception:
                append = _ORCHESTRATOR_MODE_APPEND
        return f"{base_prompt}\n\n{append.strip()}"
    if role == "subagent":
        return ""
    return settings.system_prompt


def _runtime_status_provider(
    role: AgentRole,
    subagent_runtime: object | None,
    max_sub_agents: int,
):
    if role != "orchestrator" or subagent_runtime is None:
        return None
    get_report = getattr(subagent_runtime, "get_subagent_report", None)
    if not callable(get_report):
        return None

    def provider() -> str | None:
        try:
            report = get_report()
        except Exception:  # noqa: BLE001
            return None

        workers = report.get("workers", [])
        if not isinstance(workers, list):
            return None

        lines: list[str] = []
        active_count = int(report.get("active_count", 0) or 0)
        lines.append(
            "Live worker snapshot (source of truth): "
            f"{active_count}/{max_sub_agents} active."
        )
        if not workers:
            lines.append("- No workers launched yet.")
            return "\n".join(lines)

        for index, worker in enumerate(workers[:_MAX_SNAPSHOT_WORKERS], start=1):
            if not isinstance(worker, dict):
                continue
            worker_name = str(worker.get("name", "")).strip()
            wid = str(worker.get("id", f"agent-{index}")).removeprefix("subagent-")
            status = str(worker.get("status", "unknown"))
            task = str(worker.get("task", "")).strip() or "n/a"
            name_prefix = f"{worker_name}; " if worker_name else ""
            lines.append(f"- Agent #{index} ({name_prefix}{wid}) {status}: {task}")

        extra = len(workers) - _MAX_SNAPSHOT_WORKERS
        if extra > 0:
            lines.append(f"- +{extra} more workers not shown.")
        return "\n".join(lines)

    return provider


def _active_workers_provider(role: AgentRole, subagent_runtime: object | None):
    if role != "orchestrator" or subagent_runtime is None:
        return None
    get_report = getattr(subagent_runtime, "get_subagent_report", None)
    if not callable(get_report):
        return None

    def provider() -> bool:
        try:
            report = get_report()
        except Exception:  # noqa: BLE001
            return False
        try:
            return int(report.get("active_count", 0) or 0) > 0
        except Exception:  # noqa: BLE001
            return False

    return provider


def _auto_wait_args_provider(
    role: AgentRole,
    subagent_runtime: object | None,
    wait_timeout_ms: int,
):
    if role != "orchestrator" or subagent_runtime is None:
        return None

    timeout_ms = max(int(wait_timeout_ms or 0), 0)

    def provider() -> dict[str, Any]:
        if timeout_ms <= 0:
            return {}
        return {"timeout_ms": timeout_ms}

    return provider


def _subagent_inbox_provider(
    role: AgentRole,
    subagent_coordination: SubAgentCoordinationClient | None,
):
    """Used by the LLM node auto-inject: only checks for genuinely unread
    messages (NOT pending_responses) to avoid an inject→empty-read loop."""
    if role != "subagent" or subagent_coordination is None:
        return None

    def provider() -> bool:
        try:
            state = subagent_coordination.has_unread_messages()
        except Exception:  # noqa: BLE001
            return False
        return bool(state.get("ok") and state.get("has_unread"))

    return provider


def _peer_linger_provider(
    role: AgentRole,
    subagent_coordination: SubAgentCoordinationClient | None,
):
    """Returns True when the worker has active peers that might still
    need to contact it, even though there are no unread messages yet."""
    if role != "subagent" or subagent_coordination is None:
        return None

    def provider() -> bool:
        try:
            state = subagent_coordination.state()
        except Exception:  # noqa: BLE001
            return False
        worker_id = str(state.get("worker_id", "") or "")
        directory = state.get("team_directory", [])
        if not isinstance(directory, list):
            return False
        return any(
            isinstance(p, dict)
            and p.get("active")
            and str(p.get("id", "")) != worker_id
            for p in directory
        )

    return provider


def _inbox_obligations_provider(
    role: AgentRole,
    subagent_coordination: SubAgentCoordinationClient | None,
):
    if role != "subagent" or subagent_coordination is None:
        return None

    def provider() -> bool:
        try:
            result = subagent_coordination.has_inbox_obligations()
        except Exception:  # noqa: BLE001
            return False
        return bool(result.get("ok") and result.get("has_obligations"))

    return provider


def _subagent_inbox_args_provider(
    role: AgentRole,
    subagent_coordination: SubAgentCoordinationClient | None,
):
    if role != "subagent" or subagent_coordination is None:
        return None

    def provider() -> dict[str, Any]:
        return {"limit": 10, "only_unread": True}

    return provider


def build_agent_graph(
    settings: Settings,
    *,
    role: AgentRole = "default",
    subagent_runtime: object | None = None,
    subagent_coordination: SubAgentCoordinationClient | None = None,
    prompt_registry: PromptRegistry | None = None,
):
    model = build_chat_model(settings)
    tools = collect_tools(
        settings,
        role=role,
        subagent_runtime=subagent_runtime,
        subagent_coordination=subagent_coordination,
    )
    model_with_tools = model.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}
    system_prompt = _prompt_for_role(settings, role, prompt_registry)
    runtime_status_provider = _runtime_status_provider(
        role,
        subagent_runtime,
        settings.max_sub_agents,
    )
    active_workers_provider = _active_workers_provider(role, subagent_runtime)
    auto_wait_args_provider = _auto_wait_args_provider(
        role,
        subagent_runtime,
        settings.sub_agent_wait_timeout_ms,
    )
    auto_tool_name = "WaitSubagents"
    auto_wait_provider = active_workers_provider
    if role == "subagent":
        auto_tool_name = "ReadInbox"
        auto_wait_provider = _subagent_inbox_provider(role, subagent_coordination)
        auto_wait_args_provider = _subagent_inbox_args_provider(
            role,
            subagent_coordination,
        )

    builder = StateGraph(AgentState)
    builder.add_node(
        "llm",
        make_llm_node(
            model_with_tools,
            system_prompt,
            runtime_status_provider=runtime_status_provider,
            auto_wait_provider=auto_wait_provider,
            auto_wait_tool_name=auto_tool_name,
            auto_wait_args_provider=auto_wait_args_provider,
            peer_linger_provider=_peer_linger_provider(
                role, subagent_coordination,
            ),
        ),
    )
    required_first_tool_name = (
        "SubmitTaskPlan"
        if role == "subagent" and subagent_coordination is not None
        else ""
    )
    builder.add_node(
        "tools",
        make_tool_node(
            tools_by_name,
            required_first_tool_name=required_first_tool_name,
            inbox_obligations_provider=_inbox_obligations_provider(
                role, subagent_coordination,
            ),
        ),
    )
    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", should_continue, ["tools", END])
    builder.add_edge("tools", "llm")

    return builder.compile(checkpointer=MemorySaver())
