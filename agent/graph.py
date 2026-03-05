from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent.nodes import make_llm_node, make_tool_node, should_continue
from agent.state import AgentState
from agent.tools import TOOLS

DEFAULT_SYSTEM_PROMPT = (
    "You are a local coding assistant. Use tools only when needed, keep answers concise, "
    "and show your final answer clearly."
)


def build_agent_graph(
    model_name: str,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
):
    model = ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=openai_api_key,
        base_url=openai_base_url,
    )
    model_with_tools = model.bind_tools(TOOLS)
    tools_by_name = {tool.name: tool for tool in TOOLS}

    builder = StateGraph(AgentState)
    builder.add_node("llm", make_llm_node(model_with_tools, system_prompt))
    builder.add_node("tools", make_tool_node(tools_by_name))
    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", should_continue, ["tools", END])
    builder.add_edge("tools", "llm")

    return builder.compile(checkpointer=MemorySaver())
