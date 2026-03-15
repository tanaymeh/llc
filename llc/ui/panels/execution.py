from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Log, RichLog, Static

from llc.ui.panels.mission_lane import MissionLane


class ExecutionPanel(Vertical):
    def compose(self) -> ComposeResult:
        yield Static("EXECUTION", classes="panel-title")
        yield Static("REASONING", classes="subpanel-title")
        yield RichLog(
            id="execution_reasoning_log",
            max_lines=120,
            auto_scroll=True,
            wrap=True,
            highlight=False,
        )
        yield Static("TOOLS", classes="subpanel-title")
        yield Log(id="execution_tool_log", max_lines=12, auto_scroll=True, highlight=False)
        yield Static("CHAT", classes="subpanel-title")
        yield MissionLane(id="mission_lane")

    def reasoning_log(self) -> RichLog:
        return self.query_one("#execution_reasoning_log", RichLog)

    def tool_log(self) -> Log:
        return self.query_one("#execution_tool_log", Log)

    def lane(self) -> MissionLane:
        return self.query_one("#mission_lane", MissionLane)

    def append_tool_line(self, line: str) -> None:
        self.tool_log().write_line(line)

    def clear_tool_lines(self) -> None:
        self.tool_log().clear()

    def append_reasoning_line(self, line: str) -> None:
        self.reasoning_log().write(line, scroll_end=None, animate=False)

    def clear_reasoning_lines(self) -> None:
        self.reasoning_log().clear()
