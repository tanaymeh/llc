import os
import sys
import uuid
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent import build_agent_graph

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
GRAY = "\033[90m"
WHITE = "\033[97m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"


def _w(text: str, *codes: str) -> str:
    return "".join(codes) + text + RESET


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


def _format_tool_args(args: dict[str, Any]) -> str:
    if not args:
        return ""
    parts: list[str] = []
    for k, v in args.items():
        val = repr(v) if isinstance(v, str) else str(v)
        if len(val) > 60:
            val = val[:57] + "..."
        parts.append(f"{k}={val}")
    return ", ".join(parts)


def _print_tool_calls(
    chunk: Any,
    assistant_line_open: bool,
    seen_tool_call_ids: set[str],
) -> bool:
    if not isinstance(chunk, dict):
        return assistant_line_open

    for node_update in chunk.values():
        if not isinstance(node_update, dict):
            continue

        for message in node_update.get("messages", []):
            if isinstance(message, AIMessage) and message.tool_calls:
                pending = [
                    tc
                    for tc in message.tool_calls
                    if tc.get("id") not in seen_tool_call_ids
                ]
                if not pending:
                    continue

                if assistant_line_open:
                    print()
                    assistant_line_open = False

                for tc in pending:
                    seen_tool_call_ids.add(tc.get("id"))
                    name = tc.get("name", "unknown_tool")
                    args_str = _format_tool_args(tc.get("args", {}))
                    print(
                        f"  {_w('↳', GRAY)} {_w(name, GRAY, ITALIC)}"
                        f"{_w('(', GRAY)}{_w(args_str, GRAY)}{_w(')', GRAY)}"
                    )

    return assistant_line_open


def _run_turn(agent: Any, thread_id: str, user_input: str) -> None:
    assistant_line_open = False
    seen_tool_call_ids: set[str] = set()

    for mode, chunk in agent.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config={"configurable": {"thread_id": thread_id}},
        stream_mode=["messages", "updates"],
    ):
        if mode == "messages":
            message_chunk, metadata = chunk
            if metadata.get("langgraph_node") != "llm":
                continue

            text = _message_text(message_chunk.content)
            if text:
                if not assistant_line_open:
                    print(f"\n{_w('●', CYAN)} ", end="", flush=True)
                    assistant_line_open = True
                print(text, end="", flush=True)

        elif mode == "updates":
            assistant_line_open = _print_tool_calls(
                chunk=chunk,
                assistant_line_open=assistant_line_open,
                seen_tool_call_ids=seen_tool_call_ids,
            )

    if assistant_line_open:
        print()
    print()


def _banner(model_name: str) -> None:
    line = "─" * 48
    print(f"\n{_w(line, GRAY)}")
    print(f"  {_w('local-claude-code', CYAN, BOLD)}")
    print(f"  {_w(model_name, GRAY)}")
    print(f"{_w(line, GRAY)}")
    print(f"  {_w('Type', DIM)} {_w('exit', DIM, BOLD)} {_w('or', DIM)} {_w('quit', DIM, BOLD)} {_w('to end the session.', DIM)}")
    print()


def main() -> None:
    load_dotenv()
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL")

    _banner(model_name)

    try:
        agent = build_agent_graph(
            model_name=model_name,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"\n{_w('✗', RED)} {_w('Failed to initialize model.', RED, BOLD)}")
        print(
            f"  {_w('Set MODEL_NAME, OPENAI_API_KEY, and optionally OPENAI_BASE_URL', DIM)}"
            f"\n  {_w('in your environment or .env file.', DIM)}"
        )
        print(f"  {_w(str(exc), RED)}")
        return

    thread_id = f"cli-{uuid.uuid4()}"

    while True:
        try:
            user_input = input(f"{_w('❯', GREEN, BOLD)} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{_w('Goodbye.', DIM)}")
            break

        if user_input.lower() in {"exit", "quit"}:
            print(f"{_w('Goodbye.', DIM)}")
            break
        if not user_input:
            continue

        try:
            _run_turn(agent=agent, thread_id=thread_id, user_input=user_input)
        except KeyboardInterrupt:
            print(f"\n{_w('Interrupted.', YELLOW)}\n")
        except Exception as exc:  # noqa: BLE001
            print(f"\n{_w('✗', RED)} {_w(str(exc), RED)}\n")


if __name__ == "__main__":
    main()
