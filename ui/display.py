from typing import Any

from langchain_core.messages import AIMessage

from ui.colors import BOLD, CYAN, DIM, GRAY, ITALIC, style


def message_text(content: Any) -> str:
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


def format_tool_args(args: dict[str, Any]) -> str:
    if not args:
        return ""
    parts: list[str] = []
    for k, v in args.items():
        val = repr(v) if isinstance(v, str) else str(v)
        if len(val) > 60:
            val = val[:57] + "..."
        parts.append(f"{k}={val}")
    return ", ".join(parts)


def print_tool_calls(
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
                    args_str = format_tool_args(tc.get("args", {}))
                    print(
                        f"  {style('↳', GRAY)} {style(name, GRAY, ITALIC)}"
                        f"{style('(', GRAY)}{style(args_str, GRAY)}{style(')', GRAY)}"
                    )

    return assistant_line_open


def print_banner(model_name: str) -> None:
    line = "─" * 48
    print(f"\n{style(line, GRAY)}")
    print(f"  {style('local-claude-code', CYAN, BOLD)}")
    print(f"  {style(model_name, GRAY)}")
    print(f"{style(line, GRAY)}")
    print(
        f"  {style('Type', DIM)} {style('exit', DIM, BOLD)} {style('or', DIM)} "
        f"{style('quit', DIM, BOLD)} {style('to end the session.', DIM)}"
    )
    print(f"  {style('/help', DIM, BOLD)} {style('to see available commands.', DIM)}")
    print()
