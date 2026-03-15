from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import RichLog, Static

from llc.ui.state import LogCategory, LogEntry

_CATEGORY_STYLES: dict[LogCategory, str] = {
    LogCategory.SYS: "bold white",
    LogCategory.USER: "bold cyan",
    LogCategory.AGENT: "bold green",
    LogCategory.TURN: "bold magenta",
    LogCategory.TOOL: "bold yellow",
    LogCategory.CODE: "bold bright_blue",
    LogCategory.OBS: "bold bright_cyan",
    LogCategory.WARN: "bold orange3",
    LogCategory.ERR: "bold red",
}


class LogStreamPanel(Vertical):
    def compose(self) -> ComposeResult:
        yield Static("LOG", classes="panel-title")
        yield RichLog(id="log_stream", max_lines=500, auto_scroll=True, wrap=False)

    def log_widget(self) -> RichLog:
        return self.query_one("#log_stream", RichLog)

    def append_entry(self, entry: LogEntry) -> None:
        text = Text(f"{entry.timestamp_label} ", style="dim")
        text.append(f"{entry.category.value:<4} ", style=_CATEGORY_STYLES.get(entry.category, "bold"))
        if entry.source:
            text.append(f"{entry.source:<8} ", style="dim")
        text.append(entry.message)
        self.log_widget().write(text, scroll_end=None, animate=False)

    def clear(self) -> None:
        self.log_widget().clear()
