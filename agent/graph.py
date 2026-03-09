from langchain_openai import ChatOpenAI
from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent.nodes import make_llm_node, make_tool_node, should_continue
from agent.state import AgentState
from agent.tools import collect_tools
from config import Settings


def _is_openrouter_base_url(base_url: str | None) -> bool:
    if not base_url:
        return False
    return "openrouter.ai" in base_url.lower()


def build_chat_model(settings: Settings, model_name: str | None = None):
    selected_model = model_name or settings.model_name
    if _is_openrouter_base_url(settings.openai_base_url):
        return ChatOpenRouter(
            model=selected_model,
            temperature=0,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            stream_usage=True,
        )
    return ChatOpenAI(
        model=selected_model,
        temperature=0,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        stream_usage=True,
    )


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
