import shutil
import sys
import time
from typing import Any

from langchain_core.messages import AIMessage

from ui.colors import BOLD, CYAN, DIM, GRAY, ITALIC, style


def _stringify_reasoning(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                for key in ("text", "content", "reasoning", "reasoning_content", "summary"):
                    raw = item.get(key)
                    if raw:
                        parts.append(str(raw))
                        break
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        for key in ("text", "content", "reasoning", "reasoning_content", "summary"):
            raw = value.get(key)
            if raw:
                return str(raw)
        return ""
    return str(value)


def extract_reasoning(message_chunk: Any) -> str:
    additional_kwargs = getattr(message_chunk, "additional_kwargs", {})
    if isinstance(additional_kwargs, dict):
        for key in (
            "reasoning_content",
            "reasoning_details",
            "reasoning_summary_chunk",
            "reasoning_summary",
            "reasoning",
        ):
            reasoning_text = _stringify_reasoning(additional_kwargs.get(key))
            if reasoning_text:
                return reasoning_text

    content_blocks = getattr(message_chunk, "content_blocks", None)
    if isinstance(content_blocks, list):
        parts: list[str] = []
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type", "")).lower()
            if block_type not in {"thinking", "reasoning"}:
                continue
            raw = (
                block.get("reasoning")
                or block.get("thinking")
                or block.get("reasoning_content")
                or block.get("text")
                or block.get("content")
            )
            reasoning_text = _stringify_reasoning(raw)
            if reasoning_text:
                parts.append(reasoning_text)
        if parts:
            return "\n".join(parts)

    content = getattr(message_chunk, "content", None)
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type", "")).lower()
            if block_type not in {"thinking", "reasoning"}:
                continue

            raw = (
                block.get("thinking")
                or block.get("reasoning_content")
                or block.get("reasoning")
                or block.get("text")
                or block.get("content")
            )
            reasoning_text = _stringify_reasoning(raw)
            if reasoning_text:
                parts.append(reasoning_text)
        return "\n".join(part for part in parts if part)
    return ""


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return _format_duration_unit(seconds, "second")
    if seconds < 3600:
        return _format_duration_unit(seconds / 60, "minute")
    return _format_duration_unit(seconds / 3600, "hour")


def _format_duration_unit(value: float, unit: str) -> str:
    if value >= 10:
        text = f"{value:.0f}"
    else:
        text = f"{value:.1f}"
    if text.endswith(".0"):
        text = text[:-2]
    suffix = unit if text == "1" else f"{unit}s"
    return f"{text} {suffix}"


def _truncate_to_terminal_width(text: str, prefix_width: int = 2) -> str:
    columns = shutil.get_terminal_size(fallback=(80, 24)).columns
    max_width = max(columns - prefix_width, 20)
    if len(text) <= max_width:
        return text
    if max_width <= 3:
        return text[:max_width]
    return text[: max_width - 3] + "..."


class ReasoningTracker:
    def __init__(self) -> None:
        self._started_at: float | None = None
        self._active = False
        self._line_interval_seconds = 1.0
        self._pending_tokens: list[str] = []
        self._last_emitted_at: float | None = None

    @property
    def is_active(self) -> bool:
        return self._active

    def feed(self, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return

        if self._started_at is None:
            self._started_at = time.monotonic()
        self._active = True

        tokens = self._tokenize(cleaned)
        if not tokens:
            return
        self._pending_tokens.extend(tokens)
        self._emit_if_ready(now=time.monotonic())

    def _tokenize(self, text: str) -> list[str]:
        return [tok for tok in text.split() if tok]

    def _tokens_per_line(self) -> int:
        return 20

    def _emit_if_ready(self, now: float) -> None:
        tokens_per_line = self._tokens_per_line()
        if len(self._pending_tokens) < tokens_per_line:
            return

        if self._last_emitted_at is not None:
            elapsed_since_emit = now - self._last_emitted_at
            if elapsed_since_emit < self._line_interval_seconds:
                return

        line_tokens = self._pending_tokens[:tokens_per_line]
        self._pending_tokens = self._pending_tokens[tokens_per_line:]
        self._show_line(" ".join(line_tokens))
        self._last_emitted_at = now

    def finish(self) -> None:
        if not self._active:
            return

        elapsed = 0.0
        if self._started_at is not None:
            elapsed = max(time.monotonic() - self._started_at, 0.0)

        self._clear_line()
        print(
            f"  {style(f'Reasoned for {_format_duration(elapsed)}', GRAY)}",
            flush=True,
        )
        self._started_at = None
        self._active = False
        self._pending_tokens = []
        self._last_emitted_at = None

    def _show_line(self, text: str) -> None:
        self._clear_line()
        display_text = _truncate_to_terminal_width(text)
        sys.stdout.write(f"  {style(display_text, GRAY)}")
        sys.stdout.flush()

    @staticmethod
    def _clear_line() -> None:
        sys.stdout.write("\033[2K\r")
        sys.stdout.flush()


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                block_type = str(block.get("type", "")).lower()
                if block_type in {"thinking", "reasoning"}:
                    continue
                raw = (
                    block.get("text")
                    or block.get("content")
                    or block.get("output_text")
                )
                if raw:
                    text_parts.append(str(raw))
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
