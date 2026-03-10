from __future__ import annotations

import re

from rich import box
from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def render_side_by_side_diff(diff_text: str) -> RenderableType:
    table = Table(
        box=box.SQUARE,
        expand=True,
        show_header=True,
        header_style="bold",
        border_style="grey50",
        show_lines=True,
        pad_edge=False,
        collapse_padding=True,
    )
    old_header = Text()
    old_header.append("--- ", style="bold red")
    old_header.append("Old", style="bold")
    new_header = Text()
    new_header.append("+++ ", style="bold green")
    new_header.append("New", style="bold")
    table.add_column(old_header, ratio=1, overflow="fold")
    table.add_column(new_header, ratio=1, overflow="fold")
    has_diff_lines = False
    pending_old_file = ""
    pending_old_lines: list[str] = []
    pending_new_lines: list[str] = []

    def flush_change_block() -> None:
        nonlocal has_diff_lines
        if not pending_old_lines and not pending_new_lines:
            return
        has_diff_lines = True
        for idx in range(max(len(pending_old_lines), len(pending_new_lines))):
            old_line = pending_old_lines[idx] if idx < len(pending_old_lines) else ""
            new_line = pending_new_lines[idx] if idx < len(pending_new_lines) else ""
            old_cell: RenderableType = (
                Text(f"- {old_line}", style="red")
                if old_line
                else Text("", style="dim")
            )
            new_cell: RenderableType = (
                Text(f"+ {new_line}", style="green")
                if new_line
                else Text("", style="dim")
            )
            table.add_row(old_cell, new_cell)
        pending_old_lines.clear()
        pending_new_lines.clear()

    for raw_line in diff_text.splitlines():
        line = _ANSI_RE.sub("", raw_line)
        if not line:
            flush_change_block()
            table.add_row(Text("", style="dim"), Text("", style="dim"))
            continue

        if line.startswith("=== "):
            flush_change_block()
            title = Text(line, style="bold magenta")
            table.add_row(title, title.copy())
            continue

        if line.startswith("diff --git "):
            flush_change_block()
            has_diff_lines = True
            header = Text(line, style="bold cyan")
            table.add_row(header, header.copy())
            continue

        if line.startswith("index "):
            flush_change_block()
            has_diff_lines = True
            meta = Text(line, style="cyan")
            table.add_row(meta, meta.copy())
            continue

        if line.startswith("--- "):
            flush_change_block()
            pending_old_file = line
            continue

        if line.startswith("+++ "):
            flush_change_block()
            has_diff_lines = True
            old_meta = Text(pending_old_file or "---", style="bold red")
            new_meta = Text(line, style="bold green")
            table.add_row(old_meta, new_meta)
            pending_old_file = ""
            continue

        if line.startswith("@@ "):
            flush_change_block()
            has_diff_lines = True
            hunk = Text(line, style="bold yellow")
            table.add_row(hunk, hunk.copy())
            continue

        if line.startswith("-") and not line.startswith("--- "):
            pending_old_lines.append(line[1:])
            continue

        if line.startswith("+") and not line.startswith("+++ "):
            pending_new_lines.append(line[1:])
            continue

        if line.startswith(" "):
            flush_change_block()
            common = Text(f"  {line[1:]}", style="dim")
            table.add_row(common, common.copy())
            continue

        flush_change_block()
        plain = Text(line, style="dim")
        table.add_row(plain, plain.copy())

    flush_change_block()

    if not has_diff_lines:
        return Text(diff_text)

    return table


def render_user_facing_tool_output(
    tool_name: str,
    render_mode: str,
    content: str,
) -> RenderableType:
    if render_mode == "side_by_side_diff":
        title = Text(f"{tool_name} ")
        title.append("--- old", style="bold red")
        title.append(" | ", style="dim")
        title.append("+++ new", style="bold green")
        body = render_side_by_side_diff(content)
        subtitle = Text("session baseline -> current state", style="dim")
    else:
        title = f"{tool_name} output"
        body = Text(content)
        subtitle = None
    return Panel(
        body,
        title=title,
        subtitle=subtitle,
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
