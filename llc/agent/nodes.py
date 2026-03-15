import uuid
from typing import Any, Callable, Literal

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import END

from llc.agent.state import AgentState

LlmNode = Callable[[AgentState], dict[str, Any]]
ToolNode = Callable[[AgentState], dict[str, Any]]
RuntimeStatusProvider = Callable[[], str | None]
AutoWaitProvider = Callable[[], bool]
AutoWaitArgsProvider = Callable[[], dict[str, Any]]


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


def make_llm_node(
    model_with_tools: Any,
    system_prompt: str,
    runtime_status_provider: RuntimeStatusProvider | None = None,
    auto_wait_provider: AutoWaitProvider | None = None,
    auto_wait_tool_name: str = "",
    auto_wait_args_provider: AutoWaitArgsProvider | None = None,
) -> LlmNode:
    def llm_node(state: AgentState) -> dict[str, Any]:
        prompt = system_prompt
        if runtime_status_provider is not None:
            runtime_status = runtime_status_provider()
            if runtime_status:
                prompt = f"{system_prompt.rstrip()}\n\n{runtime_status.strip()}"
        response = model_with_tools.invoke(
            [SystemMessage(content=prompt), *state["messages"]]
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
                response = _inject_auto_tool_call(
                    response,
                    tool_name=auto_wait_tool_name,
                    args=auto_wait_args,
                )
        return {"messages": [response]}

    return llm_node


def make_tool_node(tools_by_name: dict[str, Any]) -> ToolNode:
    def tool_node(state: AgentState) -> dict[str, Any]:
        last_message = state["messages"][-1]
        tool_messages: list[ToolMessage] = []

        for tool_call in getattr(last_message, "tool_calls", []):
            tool_name = tool_call["name"]
            tool = tools_by_name.get(tool_name)
            user_facing = False
            render_mode = ""
            if tool is None:
                observation = f"Unknown tool: {tool_name}"
            else:
                metadata = getattr(tool, "metadata", {}) or {}
                user_facing = bool(metadata.get("user_facing"))
                render_mode = str(metadata.get("render_mode", "") or "")
                try:
                    observation = tool.invoke(tool_call["args"])
                except Exception as exc:  # noqa: BLE001
                    observation = f"Tool '{tool_name}' failed: {exc}"

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
