from __future__ import annotations

from rich.console import RenderableType
from rich.panel import Panel
from rich.text import Text

def render_user_facing_tool_output(
    tool_name: str,
    render_mode: str,
    content: str,
) -> RenderableType:
    title = f"{tool_name} output"
    if render_mode and render_mode != "plain":
        title = f"{title} ({render_mode})"
    return Panel(
        Text(content),
        title=title,
        border_style="grey62",
    )


def format_duration(seconds: float) -> str:
    if seconds < 60:
        v, u = seconds, "second"
    elif seconds < 3600:
        v, u = seconds / 60, "minute"
    else:
        v, u = seconds / 3600, "hour"
    text = f"{v:.0f}" if v >= 10 else f"{v:.1f}"
    if text.endswith(".0"):
        text = text[:-2]
    suffix = u if text == "1" else f"{u}s"
    return f"{text} {suffix}"
