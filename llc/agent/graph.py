from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from llc.agent.llm import build_chat_model
from llc.agent.nodes import make_llm_node, make_tool_node, should_continue
from llc.agent.state import AgentState
from llc.agent.tools import collect_tools
from llc.config import Settings


def build_agent_graph(settings: Settings):
    model = build_chat_model(settings)
    tools = collect_tools(settings)
    model_with_tools = model.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}

    builder = StateGraph(AgentState)
    builder.add_node("llm", make_llm_node(model_with_tools, settings.system_prompt))
    builder.add_node("tools", make_tool_node(tools_by_name))
    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", should_continue, ["tools", END])
    builder.add_edge("tools", "llm")

    return builder.compile(checkpointer=MemorySaver())
