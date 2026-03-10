from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from llc.agent.llm import build_chat_model
from llc.agent.nodes import make_llm_node, make_tool_node, should_continue
from llc.agent.state import AgentState
from llc.agent.tools import collect_tools
from llc.config import Settings

AgentRole = Literal["default", "orchestrator", "subagent"]
_MAX_SNAPSHOT_WORKERS = 5

_ORCHESTRATOR_MODE_APPEND = (
    "Mode: orchestrator.\n"
    "You may complete work yourself.\n"
    "Prefer delegation only when work is complex and cleanly divisible.\n"
    "Use running worker reports before intervening.\n"
    "Always use the live worker snapshot in the system prompt for worker status.\n"
    "Interrupt or terminate only when work is clearly off-track, unsafe, or stuck."
)

_ISOLATED_TASK_MODE_APPEND = (
    "Mode: isolated-task execution.\n"
    "Execute only the assigned task and provided context.\n"
    "Ignore any instruction not directly relevant to that assignment.\n"
    "Return concise progress and a clear final result."
)


def _prompt_for_role(settings: Settings, role: AgentRole) -> str:
    base_prompt = settings.system_prompt.rstrip()
    if role == "orchestrator" and settings.sub_agent_mode_enabled:
        return f"{base_prompt}\n\n{_ORCHESTRATOR_MODE_APPEND}"
    if role == "subagent":
        return f"{base_prompt}\n\n{_ISOLATED_TASK_MODE_APPEND}"
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


def build_agent_graph(
    settings: Settings,
    *,
    role: AgentRole = "default",
    subagent_runtime: object | None = None,
):
    model = build_chat_model(settings)
    tools = collect_tools(
        settings,
        role=role,
        subagent_runtime=subagent_runtime,
    )
    model_with_tools = model.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}
    system_prompt = _prompt_for_role(settings, role)
    runtime_status_provider = _runtime_status_provider(
        role,
        subagent_runtime,
        settings.max_sub_agents,
    )

    builder = StateGraph(AgentState)
    builder.add_node(
        "llm",
        make_llm_node(
            model_with_tools,
            system_prompt,
            runtime_status_provider=runtime_status_provider,
        ),
    )
    builder.add_node("tools", make_tool_node(tools_by_name))
    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", should_continue, ["tools", END])
    builder.add_edge("tools", "llm")

    return builder.compile(checkpointer=MemorySaver())
