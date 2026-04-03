from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from llc.agent.llm import build_chat_model
from llc.agent.nodes import (
    ToolResultObserver,
    make_llm_node,
    make_tool_node,
    should_continue,
)
from llc.agent.state import AgentState
from llc.agent.tools import collect_tools
from llc.config import Settings

if TYPE_CHECKING:
    from llc.service.prompt_registry import PromptRegistry

AgentRole = Literal["default", "orchestrator", "subagent"]
_MAX_SNAPSHOT_WORKERS = 5
ForcedToolCallProvider = Callable[[AgentState], dict[str, Any] | None]
ToolInvocationObserver = Callable[[str], None]
_LOGGER = logging.getLogger("llc.subagents.graph")
_MUTATION_TOOL_NAMES = {"Write", "Edit", "MultiEdit", "code_grep"}

_ORCHESTRATOR_MODE_APPEND = (
    "Mode: orchestrator.\n"
    "You may complete work yourself.\n"
    "Prefer delegation only when work is complex and cleanly divisible.\n"
    "Use running worker reports before intervening.\n"
    "If active workers exist, keep this response open until they finish.\n"
    "Do not ask the user to poll for worker updates.\n"
    "Always use the live worker snapshot in the system prompt for worker status.\n"
    "When calling LaunchSubagent, always set a short, descriptive `name`.\n"
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


def _forced_tool_call_provider(
    role: AgentRole,
    subagent_coordination: object | None,
    subagent_id: str,
) -> ForcedToolCallProvider | None:
    if role != "subagent" or subagent_coordination is None:
        return None
    clean_subagent_id = subagent_id.strip()
    if not clean_subagent_id:
        return None
    getter = getattr(subagent_coordination, "next_forced_tool_call", None)
    if not callable(getter):
        return None

    def provider(_: AgentState) -> dict[str, Any] | None:
        try:
            payload = getter(clean_subagent_id)
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(payload, dict):
            return None
        tool_name = str(payload.get("tool_name", "")).strip()
        args = payload.get("args", {})
        if not tool_name:
            return None
        if not isinstance(args, dict):
            args = {}
        return {"tool_name": tool_name, "args": args}

    return provider


def _tool_invocation_observer(
    role: AgentRole,
    subagent_coordination: object | None,
    subagent_id: str,
    debug_logging_enabled: bool,
) -> ToolInvocationObserver | None:
    if role != "subagent" or subagent_coordination is None:
        return None
    clean_subagent_id = subagent_id.strip()
    if not clean_subagent_id:
        return None
    recorder = getattr(subagent_coordination, "record_tool_invocation", None)
    if not callable(recorder):
        return None

    def observer(tool_name: str) -> None:
        try:
            recorder(clean_subagent_id, tool_name)
        except Exception as exc:
            if debug_logging_enabled:
                _LOGGER.warning(
                    "[subagent-debug] event=tool_observer_error subagent_id=%s tool_name=%s error=%s",
                    clean_subagent_id,
                    tool_name,
                    str(exc),
                )
            return

    return observer


def _mutation_target_path(tool_name: str, args: dict[str, Any]) -> str:
    clean_tool = tool_name.strip()
    if clean_tool in {"Write", "Edit", "MultiEdit"}:
        raw = args.get("file_path")
    elif clean_tool == "code_grep" and bool(args.get("apply_rewrite", False)):
        raw = args.get("path")
    else:
        raw = ""
    return str(raw or "").strip()


def _is_successful_mutation_result(
    tool_name: str,
    args: dict[str, Any],
    observation: Any,
) -> bool:
    clean_tool = tool_name.strip()
    if clean_tool not in _MUTATION_TOOL_NAMES:
        return False
    if not _mutation_target_path(clean_tool, args):
        return False

    text = str(observation or "").strip()
    if not text:
        return False

    if clean_tool == "Write":
        return text.startswith("Wrote ")
    if clean_tool == "Edit":
        return text.startswith("Successfully edited ") or text.startswith(
            "Created new file "
        )
    if clean_tool == "MultiEdit":
        return text.startswith("Successfully applied ")
    if clean_tool == "code_grep":
        if not bool(args.get("apply_rewrite", False)):
            return False
        lowered = text.lower()
        if lowered.startswith("error:") or lowered.startswith("rewrite blocked:"):
            return False
        if "[preview]" in lowered or "no matches" in lowered:
            return False
        return True
    return False


def _tool_result_observer(
    role: AgentRole,
    subagent_id: str,
    debug_logging_enabled: bool,
) -> ToolResultObserver | None:
    if role != "subagent" or not debug_logging_enabled:
        return None
    clean_subagent_id = subagent_id.strip()
    if not clean_subagent_id:
        return None

    def observer(tool_name: str, args: dict[str, Any], observation: Any) -> None:
        if not _is_successful_mutation_result(tool_name, args, observation):
            return
        _LOGGER.info(
            "[subagent-debug] event=file_mutation_applied subagent_id=%s tool_name=%s file_path=%s",
            clean_subagent_id,
            tool_name.strip() or "unknown",
            _mutation_target_path(tool_name, args),
        )

    return observer


def build_agent_graph(
    settings: Settings,
    *,
    role: AgentRole = "default",
    subagent_runtime: object | None = None,
    prompt_registry: PromptRegistry | None = None,
    subagent_coordination: object | None = None,
    subagent_id: str = "",
):
    model = build_chat_model(settings)
    tools = collect_tools(
        settings,
        role=role,
        subagent_runtime=subagent_runtime,
        subagent_coordination=subagent_coordination,
        subagent_id=subagent_id,
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
    forced_tool_provider = _forced_tool_call_provider(
        role,
        subagent_coordination,
        subagent_id,
    )
    tool_invocation_observer = _tool_invocation_observer(
        role,
        subagent_coordination,
        subagent_id,
        settings.sub_agent_debug_logging,
    )
    tool_result_observer = _tool_result_observer(
        role,
        subagent_id,
        settings.sub_agent_debug_logging,
    )

    builder = StateGraph(AgentState)
    builder.add_node(
        "llm",
        make_llm_node(
            model_with_tools,
            system_prompt,
            runtime_status_provider=runtime_status_provider,
            auto_wait_provider=active_workers_provider,
            auto_wait_tool_name="WaitSubagents",
            auto_wait_args_provider=auto_wait_args_provider,
            forced_tool_call_provider=forced_tool_provider,
        ),
    )
    builder.add_node(
        "tools",
        make_tool_node(
            tools_by_name,
            tool_invocation_observer=tool_invocation_observer,
            tool_result_observer=tool_result_observer,
        ),
    )
    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", should_continue, ["tools", END])
    builder.add_edge("tools", "llm")

    return builder.compile(checkpointer=MemorySaver())
