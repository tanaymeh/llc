import os
import uuid
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent import build_agent_graph


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                text_parts.append(block)
        return "\n".join(part for part in text_parts if part)
    return str(content)


def _run_turn(agent: Any, thread_id: str, user_input: str) -> None:
    seen_message_ids: set[str] = set()

    for chunk in agent.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="values",
    ):
        messages = chunk.get("messages", [])
        if not messages:
            continue

        message = messages[-1]
        message_id = str(
            getattr(message, "id", "")
            or f"{message.__class__.__name__}:{len(messages)}:{hash(str(message.content))}"
        )

        if message_id in seen_message_ids:
            continue
        seen_message_ids.add(message_id)

        if isinstance(message, AIMessage) and message.tool_calls:
            print("\n[tool calls]")
            for tool_call in message.tool_calls:
                tool_name = tool_call.get("name", "unknown_tool")
                tool_args = tool_call.get("args", {})
                print(f"- {tool_name}({tool_args})")
        elif isinstance(message, ToolMessage):
            print(f"\n[tool result]\n{message.content}")
        elif isinstance(message, AIMessage):
            text = _message_text(message.content).strip()
            if text:
                print(f"\nAssistant: {text}")


def main() -> None:
    load_dotenv()
    model_name = os.getenv("MODEL_NAME", "openai:gpt-4o-mini")

    print(f"Starting LangGraph coding agent with model: {model_name}")
    print("Type 'exit' or 'quit' to end.\n")

    try:
        agent = build_agent_graph(model_name=model_name)
    except Exception as exc:  # noqa: BLE001
        print("\nFailed to initialize model.")
        print(
            "Set MODEL_NAME and matching API key in your environment or .env file "
            "(OPENAI_API_KEY or ANTHROPIC_API_KEY)."
        )
        print(f"Details: {exc}")
        return
    thread_id = f"cli-{uuid.uuid4()}"

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("Exiting.")
            break
        if not user_input:
            continue

        try:
            _run_turn(agent=agent, thread_id=thread_id, user_input=user_input)
        except Exception as exc:  # noqa: BLE001
            print(f"\n[error] {exc}")


if __name__ == "__main__":
    main()
