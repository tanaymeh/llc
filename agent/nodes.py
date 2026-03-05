from typing import Any, Callable, Literal

from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import END

from agent.state import AgentState

LlmNode = Callable[[AgentState], dict[str, Any]]
ToolNode = Callable[[AgentState], dict[str, Any]]


def make_llm_node(model_with_tools: Any, system_prompt: str) -> LlmNode:
    def llm_node(state: AgentState) -> dict[str, Any]:
        response = model_with_tools.invoke(
            [SystemMessage(content=system_prompt), *state["messages"]]
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
            if tool is None:
                observation = f"Unknown tool: {tool_name}"
            else:
                try:
                    observation = tool.invoke(tool_call["args"])
                except Exception as exc:  # noqa: BLE001
                    observation = f"Tool '{tool_name}' failed: {exc}"

            tool_messages.append(
                ToolMessage(
                    content=str(observation),
                    tool_call_id=tool_call["id"],
                )
            )

        return {"messages": tool_messages}

    return tool_node


def should_continue(state: AgentState) -> Literal["tools", END]:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END
