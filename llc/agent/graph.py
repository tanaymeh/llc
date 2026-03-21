from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from llc.agent.llm import build_chat_model
from llc.agent.nodes import make_llm_node, make_tool_node, should_continue
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
    "Interrupt or terminate only when work is clearly off-track, unsafe, or stuck."
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
            wid = str(worker.get("id", f"agent-{index}")).removeprefix("subagent-")
            status = str(worker.get("status", "unknown"))
            task = str(worker.get("task", "")).strip() or "n/a"
            lines.append(f"- Agent #{index} ({wid}) {status}: {task}")

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


def build_agent_graph(
    settings: Settings,
    *,
    role: AgentRole = "default",
    subagent_runtime: object | None = None,
    prompt_registry: PromptRegistry | None = None,
):
    model = build_chat_model(settings)
    tools = collect_tools(
        settings,
        role=role,
        subagent_runtime=subagent_runtime,
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

    builder = StateGraph(AgentState)
    builder.add_node(
        "llm",
        make_llm_node(
            model_with_tools,
            system_prompt,
            runtime_status_provider=runtime_status_provider,
            auto_wait_provider=active_workers_provider,
            auto_wait_tool_name="WaitSubagents",
        ),
    )
    builder.add_node("tools", make_tool_node(tools_by_name))
    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", should_continue, ["tools", END])
    builder.add_edge("tools", "llm")

    return builder.compile(checkpointer=MemorySaver())
