import json
import uuid
from typing import Any, Callable, Coroutine, Literal

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import END

from llc.observability import start_child_span, update_observation
from llc.agent.state import AgentState

LlmNode = Callable[[AgentState], dict[str, Any]]
ToolNode = Callable[[AgentState], Coroutine[Any, Any, dict[str, Any]]]
RuntimeStatusProvider = Callable[[], str | None]
AutoWaitProvider = Callable[[], bool]
AutoWaitArgsProvider = Callable[[], dict[str, Any]]
InboxObligationsProvider = Callable[[], bool]
_MAX_REPORT_POLLS_BEFORE_AUTO_WAIT = 3
_AUTO_WAIT_TIMEOUT_MS = 1200
_DONE_REPORT_HINT = (
    "Auto-guard: all sub-agents are already done. "
    "Stop polling reports and provide a final combined outcome now."
)
_MAX_TOOL_OUTPUT_PREVIEW_CHARS = 600
_INBOX_TOOL_NAMES = frozenset({
    "ReadInbox", "SendPeerMessage", "HasInboxMessages", "WaitForPeerMessage",
})
_INBOX_BLOCKED_HINT = (
    "BLOCKED: You have unread peer messages or unanswered questions from peers. "
    "You MUST call ReadInbox first, then respond to each message with "
    "SendPeerMessage(kind='response') before using any other tool."
)
_MAX_CONSECUTIVE_INBOX_BLOCKS = 4


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
) -> Any:
    if not isinstance(response, AIMessage):
        return response
    if getattr(response, "tool_calls", None):
        return response
    tool_call = {
        "id": f"auto-wait-{uuid.uuid4().hex[:12]}",
        "type": "tool_call",
        "name": tool_name,
        "args": args,
    }
    model_copy = getattr(response, "model_copy", None)
    if callable(model_copy):
        return model_copy(update={"tool_calls": [tool_call]})
    return AIMessage(content=response.content, tool_calls=[tool_call])


_PEER_LINGER_MAX_ATTEMPTS = 3
_PEER_LINGER_WAIT_MS = 15000


def make_llm_node(
    model_with_tools: Any,
    system_prompt: str,
    runtime_status_provider: RuntimeStatusProvider | None = None,
    auto_wait_provider: AutoWaitProvider | None = None,
    auto_wait_tool_name: str = "",
    auto_wait_args_provider: AutoWaitArgsProvider | None = None,
    peer_linger_provider: AutoWaitProvider | None = None,
) -> LlmNode:
    peer_linger_attempts = 0

    def llm_node(state: AgentState) -> dict[str, Any]:
        nonlocal peer_linger_attempts
        prompt = system_prompt
        if runtime_status_provider is not None:
            runtime_status = runtime_status_provider()
            if runtime_status:
                prompt = f"{system_prompt.rstrip()}\n\n{runtime_status.strip()}"
        messages = list(state["messages"])
        if prompt.strip():
            messages = [SystemMessage(content=prompt), *messages]
        response = model_with_tools.invoke(messages)
        if auto_wait_provider is not None and auto_wait_tool_name:
            should_wait = False
            try:
                should_wait = auto_wait_provider()
            except Exception:  # noqa: BLE001
                should_wait = False
            if should_wait:
                peer_linger_attempts = 0
                auto_wait_args: dict[str, Any] = {}
                if auto_wait_args_provider is not None:
                    try:
                        candidate_args = auto_wait_args_provider()
                    except Exception:  # noqa: BLE001
                        candidate_args = {}
                    if isinstance(candidate_args, dict):
                        auto_wait_args = candidate_args
                response = _inject_auto_tool_call(
                    response,
                    tool_name=auto_wait_tool_name,
                    args=auto_wait_args,
                )
            elif (
                peer_linger_provider is not None
                and peer_linger_attempts < _PEER_LINGER_MAX_ATTEMPTS
            ):
                should_linger = False
                try:
                    should_linger = peer_linger_provider()
                except Exception:  # noqa: BLE001
                    should_linger = False
                if should_linger:
                    peer_linger_attempts += 1
                    response = _inject_auto_tool_call(
                        response,
                        tool_name="WaitForPeerMessage",
                        args={"timeout_ms": _PEER_LINGER_WAIT_MS},
                    )
        return {"messages": [response]}

    return llm_node


def make_tool_node(
    tools_by_name: dict[str, Any],
    *,
    required_first_tool_name: str = "",
    inbox_obligations_provider: InboxObligationsProvider | None = None,
) -> ToolNode:
    report_poll_streak = 0
    required_tool_done = not bool(required_first_tool_name.strip())
    required_first_tool_clean = required_first_tool_name.strip()
    required_first_tool_hint = (
        f"Tool usage is blocked until `{required_first_tool_clean}` succeeds "
        "with a multi-step plan (at least 2 steps)."
        if required_first_tool_clean
        else ""
    )

    consecutive_inbox_blocks = 0

    def _check_inbox_obligations() -> bool:
        if inbox_obligations_provider is None:
            return False
        try:
            return inbox_obligations_provider()
        except Exception:  # noqa: BLE001
            return False

    async def tool_node(state: AgentState) -> dict[str, Any]:
        nonlocal report_poll_streak
        nonlocal required_tool_done
        nonlocal consecutive_inbox_blocks
        last_message = state["messages"][-1]
        tool_messages: list[ToolMessage] = []

        tool_calls = getattr(last_message, "tool_calls", [])
        has_obligations = _check_inbox_obligations()
        requested_names = {tc["name"] for tc in tool_calls}
        inbox_blocked = (
            has_obligations
            and required_tool_done
            and not (requested_names & _INBOX_TOOL_NAMES)
            and consecutive_inbox_blocks < _MAX_CONSECUTIVE_INBOX_BLOCKS
        )
        if inbox_blocked:
            consecutive_inbox_blocks += 1
        elif requested_names & _INBOX_TOOL_NAMES:
            consecutive_inbox_blocks = 0

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
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
            ) as tool_span:
                if tool is None:
                    report_poll_streak = 0
                    observation = f"Unknown tool: {tool_name}"
                    update_observation(
                        tool_span,
                        output={"status": "unknown_tool", "tool_name": tool_name},
                    )
                elif (
                    not required_tool_done
                    and required_first_tool_clean
                    and tool_name != required_first_tool_clean
                ):
                    report_poll_streak = 0
                    observation = (
                        f"Tool '{tool_name}' blocked. {required_first_tool_hint}"
                    )
                    update_observation(
                        tool_span,
                        output={
                            "status": "blocked_until_plan",
                            "tool_name": tool_name,
                            "required_tool": required_first_tool_clean,
                        },
                    )
                elif (
                    inbox_blocked
                    and tool_name not in _INBOX_TOOL_NAMES
                ):
                    report_poll_streak = 0
                    observation = f"Tool '{tool_name}' blocked. {_INBOX_BLOCKED_HINT}"
                    update_observation(
                        tool_span,
                        output={
                            "status": "blocked_inbox_obligations",
                            "tool_name": tool_name,
                        },
                    )
                else:
                    metadata = getattr(tool, "metadata", {}) or {}
                    user_facing = bool(metadata.get("user_facing"))
                    render_mode = str(metadata.get("render_mode", "") or "")
                    try:
                        observation = await tool.ainvoke(tool_call["args"])
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
                                        wait_observation = await wait_tool.ainvoke(
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
                        if (
                            not required_tool_done
                            and required_first_tool_clean
                            and tool_name == required_first_tool_clean
                        ):
                            plan_payload = _parse_report_payload(observation)
                            ok = bool(
                                isinstance(plan_payload, dict)
                                and plan_payload.get("ok")
                            )
                            step_count = 0
                            if isinstance(plan_payload, dict):
                                try:
                                    step_count = int(plan_payload.get("step_count", 0) or 0)
                                except Exception:  # noqa: BLE001
                                    step_count = 0
                            if ok and step_count >= 2:
                                required_tool_done = True
                            else:
                                observation = (
                                    f"{observation}\n\n"
                                    f"{required_first_tool_hint}"
                                )
                        update_observation(
                            tool_span,
                            output={
                                "status": "ok",
                                "tool_name": tool_name,
                                "result_preview": _tool_output_preview(observation),
                            },
                        )

            tool_messages.append(
                ToolMessage(
                    content=str(observation),
                    tool_call_id=tool_call["id"],
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
