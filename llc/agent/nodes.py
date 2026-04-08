import json
import uuid
from typing import Any, Callable, Literal

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import END

from llc.agent.llm_retry import invoke_with_retry
from llc.observability import start_child_span, update_observation
from llc.agent.state import AgentState

LlmNode = Callable[[AgentState], dict[str, Any]]
ToolNode = Callable[[AgentState], dict[str, Any]]
RuntimeStatusProvider = Callable[[], str | None]
AutoWaitProvider = Callable[[], bool]
AutoWaitArgsProvider = Callable[[], dict[str, Any]]
ForcedToolCallProvider = Callable[[AgentState], dict[str, Any] | None]
ToolInvocationObserver = Callable[[str], None]
ToolResultObserver = Callable[[str, dict[str, Any], Any], None]
_FORCED_TOOL_CALL_ID_PREFIX = "forced-tool-"
_AUTO_WAIT_TOOL_CALL_ID_PREFIX = "auto-wait-"
_FORCED_COORDINATION_TOOL_MESSAGES = {
    "ReadInbox": "you have unread teammate message(s); review and respond with priority.",
    "ReviewHeldLocks": "you hold file lock lease(s); decide KEEP or RELEASE to avoid stale ownership.",
}
_MAX_REPORT_POLLS_BEFORE_AUTO_WAIT = 3
_AUTO_WAIT_TIMEOUT_MS = 1200
_DONE_REPORT_HINT = (
    "Auto-guard: all sub-agents are already done. "
    "Stop polling reports and provide a final combined outcome now."
)
_MAX_TOOL_OUTPUT_PREVIEW_CHARS = 600


def _tool_output_preview(value: Any) -> str:
    text = str(value).strip()
    if len(text) <= _MAX_TOOL_OUTPUT_PREVIEW_CHARS:
        return text
    return text[: _MAX_TOOL_OUTPUT_PREVIEW_CHARS - 3] + "..."


def _parse_report_payload(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, dict):
        return payload
    if not isinstance(payload, str):
        return None
    try:
        parsed = json.loads(payload)
    except ValueError:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _inject_auto_tool_call(
    response: Any,
    *,
    tool_name: str,
    args: dict[str, Any],
    call_id_prefix: str,
) -> Any:
    if not isinstance(response, AIMessage):
        return response
    existing_calls = list(getattr(response, "tool_calls", None) or [])
    if existing_calls:
        first_call = existing_calls[0]
        first_name = str(first_call.get("name", "")).strip() if isinstance(first_call, dict) else ""
        if first_name == tool_name:
            return response
    tool_call = {
        "id": f"{call_id_prefix}-{uuid.uuid4().hex[:12]}",
        "type": "tool_call",
        "name": tool_name,
        "args": args,
    }
    merged_calls = [tool_call, *existing_calls]
    model_copy = getattr(response, "model_copy", None)
    if callable(model_copy):
        return model_copy(update={"tool_calls": merged_calls})
    return AIMessage(content=response.content, tool_calls=merged_calls)


def _decorate_forced_coordination_output(
    *,
    tool_call_id: str,
    tool_name: str,
    observation: Any,
) -> Any:
    if not tool_call_id.startswith(_FORCED_TOOL_CALL_ID_PREFIX):
        return observation
    system_ping_message = _FORCED_COORDINATION_TOOL_MESSAGES.get(tool_name)
    if not system_ping_message:
        return observation
    payload = _parse_report_payload(observation)
    if not isinstance(payload, dict):
        return observation
    payload = dict(payload)
    payload["message"] = f"[System Ping]: {system_ping_message}"
    payload["system_ping"] = True
    return json.dumps(payload, indent=2, sort_keys=True, default=str)


def make_llm_node(
    model_with_tools: Any,
    system_prompt: str,
    runtime_status_provider: RuntimeStatusProvider | None = None,
    auto_wait_provider: AutoWaitProvider | None = None,
    auto_wait_tool_name: str = "",
    auto_wait_args_provider: AutoWaitArgsProvider | None = None,
    forced_tool_call_provider: ForcedToolCallProvider | None = None,
) -> LlmNode:
    def llm_node(state: AgentState) -> dict[str, Any]:
        prompt = system_prompt
        if runtime_status_provider is not None:
            runtime_status = runtime_status_provider()
            if runtime_status:
                prompt = f"{system_prompt.rstrip()}\n\n{runtime_status.strip()}"
        messages = list(state["messages"])
        if prompt.strip():
            messages = [SystemMessage(content=prompt), *messages]
        response = invoke_with_retry(model_with_tools, messages)
        if forced_tool_call_provider is not None:
            forced_call: dict[str, Any] | None = None
            try:
                forced_call = forced_tool_call_provider(state)
            except Exception:  # noqa: BLE001
                forced_call = None
            if isinstance(forced_call, dict):
                tool_name = str(forced_call.get("tool_name", "")).strip()
                raw_args = forced_call.get("args", {})
                args = raw_args if isinstance(raw_args, dict) else {}
                if tool_name:
                    response = _inject_auto_tool_call(
                        response,
                        tool_name=tool_name,
                        args=args,
                        call_id_prefix=_FORCED_TOOL_CALL_ID_PREFIX,
                    )
        if auto_wait_provider is not None and auto_wait_tool_name:
            should_wait = False
            try:
                should_wait = auto_wait_provider()
            except Exception:  # noqa: BLE001
                should_wait = False
            if should_wait:
                auto_wait_args: dict[str, Any] = {}
                if auto_wait_args_provider is not None:
                    try:
                        candidate_args = auto_wait_args_provider()
                    except Exception:  # noqa: BLE001
                        candidate_args = {}
                    if isinstance(candidate_args, dict):
                        auto_wait_args = candidate_args
                if not getattr(response, "tool_calls", None):
                    response = _inject_auto_tool_call(
                        response,
                        tool_name=auto_wait_tool_name,
                        args=auto_wait_args,
                        call_id_prefix=_AUTO_WAIT_TOOL_CALL_ID_PREFIX,
                    )
        return {"messages": [response]}

    return llm_node


def make_tool_node(
    tools_by_name: dict[str, Any],
    *,
    tool_invocation_observer: ToolInvocationObserver | None = None,
    tool_result_observer: ToolResultObserver | None = None,
) -> ToolNode:
    report_poll_streak = 0

    def tool_node(state: AgentState) -> dict[str, Any]:
        nonlocal report_poll_streak
        last_message = state["messages"][-1]
        tool_messages: list[ToolMessage] = []

        for tool_call in getattr(last_message, "tool_calls", []):
            tool_name = tool_call["name"]
            tool_call_id = str(tool_call.get("id", "")).strip()
            if not tool_call_id:
                tool_call_id = f"tool-{uuid.uuid4().hex[:12]}"
            tool = tools_by_name.get(tool_name)
            user_facing = False
            render_mode = ""
            with start_child_span(
                "llc.tool.invoke",
                input_payload={
                    "tool_name": tool_name,
                    "args": tool_call.get("args", {}),
                },
                tags=("llc", "api", "tool"),
                metadata={"llc_tool_name": tool_name},
                as_type="tool",
            ) as tool_span:
                if tool is None:
                    report_poll_streak = 0
                    observation = f"Unknown tool: {tool_name}"
                    update_observation(
                        tool_span,
                        output={"status": "unknown_tool", "tool_name": tool_name},
                    )
                else:
                    metadata = getattr(tool, "metadata", {}) or {}
                    user_facing = bool(metadata.get("user_facing"))
                    render_mode = str(metadata.get("render_mode", "") or "")
                    try:
                        observation = tool.invoke(tool_call["args"])
                    except Exception as exc:  # noqa: BLE001
                        report_poll_streak = 0
                        observation = f"Tool '{tool_name}' failed: {exc}"
                        update_observation(
                            tool_span,
                            output={
                                "status": "error",
                                "tool_name": tool_name,
                                "error": str(exc),
                            },
                        )
                    else:
                        if tool_result_observer is not None:
                            raw_args = tool_call.get("args", {})
                            try:
                                tool_result_observer(
                                    tool_name,
                                    raw_args if isinstance(raw_args, dict) else {},
                                    observation,
                                )
                            except Exception:
                                pass
                        if tool_name == "GetSubagentReport":
                            report_poll_streak += 1
                            report_payload = _parse_report_payload(observation)
                            active_count = None
                            if isinstance(report_payload, dict):
                                try:
                                    active_count = int(
                                        report_payload.get("active_count", 0) or 0
                                    )
                                except Exception:  # noqa: BLE001
                                    active_count = None

                            if active_count == 0:
                                report_poll_streak = 0
                                observation = f"{observation}\n\n{_DONE_REPORT_HINT}"
                            elif report_poll_streak >= _MAX_REPORT_POLLS_BEFORE_AUTO_WAIT:
                                wait_tool = tools_by_name.get("WaitSubagents")
                                if wait_tool is not None:
                                    try:
                                        wait_observation = wait_tool.invoke(
                                            {"timeout_ms": _AUTO_WAIT_TIMEOUT_MS}
                                        )
                                    except Exception as exc:  # noqa: BLE001
                                        wait_observation = (
                                            f"Auto WaitSubagents failed: {exc}"
                                        )
                                    observation = (
                                        f"{observation}\n\n"
                                        f"Auto-guard: detected {report_poll_streak} consecutive "
                                        "GetSubagentReport calls. "
                                        "Executed WaitSubagents to avoid busy polling.\n"
                                        f"{wait_observation}"
                                    )
                                else:
                                    observation = (
                                        f"{observation}\n\n"
                                        "Auto-guard: repeated GetSubagentReport calls detected. "
                                        "Use WaitSubagents(timeout_ms=1200) before polling again."
                                    )
                                report_poll_streak = 0
                        elif tool_name == "WaitSubagents":
                            report_poll_streak = 0
                        else:
                            report_poll_streak = 0
                        observation = _decorate_forced_coordination_output(
                            tool_call_id=tool_call_id,
                            tool_name=tool_name,
                            observation=observation,
                        )
                        update_observation(
                            tool_span,
                            output={
                                "status": "ok",
                                "tool_name": tool_name,
                                "result_preview": _tool_output_preview(observation),
                            },
                        )
            if tool_invocation_observer is not None:
                try:
                    tool_invocation_observer(tool_name)
                except Exception:
                    pass

            tool_messages.append(
                ToolMessage(
                    content=str(observation),
                    tool_call_id=tool_call_id,
                    additional_kwargs={
                        "tool_name": tool_name,
                        "user_facing": user_facing,
                        "tool_render_mode": render_mode,
                    },
                )
            )

        return {"messages": tool_messages}

    return tool_node


def should_continue(state: AgentState) -> Literal["tools", END]:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END
