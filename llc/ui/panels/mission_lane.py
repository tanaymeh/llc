from __future__ import annotations

from rich.console import Group
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Markdown, Static

from llc.ui.rendering import render_user_facing_tool_output
from llc.ui.state import MessageLaneEntry


class MissionMessage(Vertical):
    def __init__(self, entry: MessageLaneEntry) -> None:
        super().__init__(classes=f"mission-message role-{entry.role}")
        self._entry = entry
        self._reasoning_dots = 1

    def compose(self) -> ComposeResult:
        yield Static("", classes="lane-header")
        yield Static("", classes="lane-meta")
        yield Markdown("", classes="lane-markdown")
        yield Static("", classes="lane-content")
        yield Static("", classes="lane-tool-output")

    async def on_mount(self) -> None:
        self.set_interval(0.4, self._tick_reasoning_indicator)
        await self.sync_content(self._entry)

    def _show_reasoning_indicator(self, entry: MessageLaneEntry) -> bool:
        return (
            entry.role == "agent"
            and entry.agent_status == "reasoning"
            and not entry.content.strip()
        )

    def _tick_reasoning_indicator(self) -> None:
        if not self._show_reasoning_indicator(self._entry):
            self._reasoning_dots = 1
            return
        self._reasoning_dots = 1 + (self._reasoning_dots % 3)
        self._render_header(self._entry)

    def _render_header(self, entry: MessageLaneEntry) -> None:
        header_text = Text(f"[{entry.timestamp_label}] {entry.title}")
        if self._show_reasoning_indicator(entry):
            header_text.append("  ")
            header_text.append(f"Reasoning{'.' * self._reasoning_dots}", style="dim")
        self.query_one(".lane-header", Static).update(header_text)

    async def sync_content(self, entry: MessageLaneEntry) -> None:
        self._entry = entry
        content = entry.content.strip()
        markdown_widget = self.query_one(".lane-markdown", Markdown)
        content_widget = self.query_one(".lane-content", Static)
        if entry.role == "agent":
            await markdown_widget.update(content)
            markdown_widget.display = bool(content)
            content_widget.update("")
            content_widget.display = False
        else:
            await markdown_widget.update("")
            markdown_widget.display = False
            content_widget.update(Text(content))
            content_widget.display = bool(content)
        self.sync_static(entry)

    def sync_static(self, entry: MessageLaneEntry) -> None:
        self._entry = entry
        self._render_header(entry)
        status = entry.agent_status if entry.agent_status != "reasoning" else ""
        meta_parts = [part for part in (entry.tool_status, status) if part]
        meta_line = " | ".join(meta_parts)
        meta_widget = self.query_one(".lane-meta", Static)
        meta_widget.update(Text(meta_line))
        meta_widget.display = bool(meta_line)
        renderables = [
            render_user_facing_tool_output(output.tool_name, output.render_mode, output.content)
            for output in entry.tool_outputs
        ]
        tool_output_widget = self.query_one(".lane-tool-output", Static)
        tool_output_widget.update(Group(*renderables) if renderables else "")
        tool_output_widget.display = bool(renderables)


class MissionLane(VerticalScroll):
    async def mount_entry(self, entry: MessageLaneEntry) -> MissionMessage:
        widget = MissionMessage(entry)
        await self.mount(widget)
        self.scroll_end(animate=False)
        return widget
