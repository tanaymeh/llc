import shutil
import sys
import time
import os
from typing import Any

from langchain_core.messages import AIMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from ui.colors import BLUE, CYAN, GRAY, LIGHT_GRAY, style
from ui.token_tracker import TurnUsage

console = Console()


def _is_light_terminal() -> bool:
    colorfgbg = os.getenv("COLORFGBG", "")
    if not colorfgbg:
        return False
    parts = [part for part in colorfgbg.split(";") if part.isdigit()]
    if not parts:
        return False
    return int(parts[-1]) >= 7


IS_LIGHT_THEME = _is_light_terminal()
ACCENT_STYLE = "blue" if IS_LIGHT_THEME else "cyan"
MUTED_STYLE = "grey42" if IS_LIGHT_THEME else "grey70"
TOOL_NAME_STYLE = "blue italic" if IS_LIGHT_THEME else "cyan italic"
MARKDOWN_CODE_THEME = "ansi_light" if IS_LIGHT_THEME else "ansi_dark"
HUD_COLOR = BLUE if IS_LIGHT_THEME else CYAN
HUD_MUTED_COLOR = GRAY if IS_LIGHT_THEME else LIGHT_GRAY

LLC_LOGO = r"""
  ██╗     ██╗      ██████╗
  ██║     ██║     ██╔════╝
  ██║     ██║     ██║
  ██║     ██║     ██║
  ███████╗███████╗╚██████╗
  ╚══════╝╚══════╝ ╚═════╝"""


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
        console.print(
            f"  [dim italic]Reasoned for {_format_duration(elapsed)}[/dim italic]"
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


def collect_new_tool_calls(
    chunk: Any,
    seen_tool_call_ids: set[str],
) -> list[dict[str, Any]]:
    if not isinstance(chunk, dict):
        return []
    pending: list[dict[str, Any]] = []
    for node_update in chunk.values():
        if not isinstance(node_update, dict):
            continue
        for message in node_update.get("messages", []):
            if isinstance(message, AIMessage) and message.tool_calls:
                for tc in message.tool_calls:
                    if tc.get("id") not in seen_tool_call_ids:
                        seen_tool_call_ids.add(tc.get("id"))
                        pending.append(tc)
    return pending


def render_assistant_markdown(text: str) -> None:
    if not text.strip():
        return
    console.print(f"[bold {ACCENT_STYLE}]●[/bold {ACCENT_STYLE}] ", end="")
    console.print(Markdown(text, code_theme=MARKDOWN_CODE_THEME))


def render_session_hud(
    total_input: int,
    total_output: int,
    total_cost: float | None = None,
) -> None:
    hud = f"Σ {total_input:,} in / {total_output:,} out"
    if total_cost is not None and total_cost > 0:
        hud += f" | ${total_cost:.4f}"

    cols = shutil.get_terminal_size(fallback=(80, 24)).columns
    if len(hud) >= cols:
        hud = hud[: max(cols - 1, 1)]
    col = max(cols - len(hud) + 1, 1)
    sys.stdout.write("\0337")
    sys.stdout.write("\033[1;1H\033[2K")
    sys.stdout.write(f"\033[1;{col}H")
    parts = hud.split("|", maxsplit=1)
    if len(parts) == 2:
        left = parts[0].rstrip()
        right = "| " + parts[1].strip()
        sys.stdout.write(style(left, HUD_COLOR))
        sys.stdout.write(" ")
        sys.stdout.write(style(right, HUD_MUTED_COLOR))
    else:
        sys.stdout.write(style(hud, HUD_COLOR))
    sys.stdout.write("\0338")
    sys.stdout.flush()


def print_usage(
    turn: TurnUsage,
    cost: float | None = None,
) -> None:
    parts = Text()
    parts.append("  tokens: ", style="dim")
    parts.append(f"{turn.input_tokens:,}", style="green")
    parts.append(" in", style="dim")
    parts.append(" / ", style="dim")
    parts.append(f"{turn.output_tokens:,}", style="yellow")
    parts.append(" out", style="dim")
    if cost is not None:
        parts.append("  │  ", style="dim")
        parts.append("cost: ", style="dim")
        parts.append(f"${cost:.4f}", style="dim italic")
    console.print(parts)


def print_session_summary(
    total_input: int,
    total_output: int,
    total_cost: float | None = None,
) -> None:
    parts = Text()
    parts.append("Session totals: ", style="bold dim")
    parts.append(f"{total_input:,}", style="green")
    parts.append(" in", style="dim")
    parts.append(" / ", style="dim")
    parts.append(f"{total_output:,}", style="yellow")
    parts.append(" out", style="dim")
    if total_cost is not None and total_cost > 0:
        parts.append("  │  ", style="dim")
        parts.append(f"${total_cost:.4f}", style="dim italic")
    console.print(parts)


def print_banner(model_name: str) -> None:
    logo = Text(LLC_LOGO, style=f"bold {ACCENT_STYLE}")
    subtitle = Text()
    subtitle.append("\n  Model: ", style=MUTED_STYLE)
    subtitle.append(model_name, style="bold")
    subtitle.append("\n  Type ", style=MUTED_STYLE)
    subtitle.append("exit", style=f"bold {MUTED_STYLE}")
    subtitle.append(" or ", style=MUTED_STYLE)
    subtitle.append("quit", style=f"bold {MUTED_STYLE}")
    subtitle.append(" to end the session.", style=MUTED_STYLE)
    subtitle.append("\n  ", style=MUTED_STYLE)
    subtitle.append("/help", style=f"bold {MUTED_STYLE}")
    subtitle.append(" to see available commands.", style=MUTED_STYLE)

    content = Text()
    content.append_text(logo)
    content.append_text(subtitle)

    panel = Panel(
        content,
        border_style=ACCENT_STYLE,
        padding=(0, 2),
    )
    console.print()
    console.print(panel)
    console.print()
