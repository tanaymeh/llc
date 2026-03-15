from __future__ import annotations

from rich.console import Group, RenderableType
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Collapsible, Input, Markdown, OptionList, Static, TextArea

from llc.models import AvailableModel


def _single_line_preview(text: str, limit: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(limit - 3, 0)] + "..."


class ModelPickerScreen(ModalScreen[str | None]):
    BINDINGS = [
        Binding("up", "move_up", show=False, priority=True),
        Binding("down", "move_down", show=False, priority=True),
        Binding("enter,return", "accept", show=False, priority=True),
        Binding("escape", "cancel", show=False, priority=True),
    ]

    CSS = """
    ModelPickerScreen {
        align: center middle;
    }
    #model_picker_box {
        width: 80;
        max-width: 90%;
        height: 24;
        max-height: 80%;
        background: $surface;
        border: round $primary;
        padding: 1 2;
    }
    #model_search {
        width: 1fr;
        margin: 0 0 1 0;
    }
    #model_list {
        width: 1fr;
        height: 1fr;
    }
    """

    def __init__(self, models: list[AvailableModel]) -> None:
        super().__init__()
        self._models = models

    def compose(self) -> ComposeResult:
        with Vertical(id="model_picker_box"):
            yield Input(placeholder="Search models…", id="model_search")
            yield OptionList(id="model_list")

    async def on_mount(self) -> None:
        self._populate("")
        self.query_one("#model_search", Input).focus()

    @on(Input.Changed, "#model_search")
    def _on_search(self, event: Input.Changed) -> None:
        self._populate(event.value.strip().lower())

    def _populate(self, query: str) -> None:
        option_list = self.query_one("#model_list", OptionList)
        option_list.clear_options()
        for m in self._models:
            if query and query not in m.id.lower() and query not in m.name.lower():
                continue
            option_list.add_option(m.id)
        if option_list.option_count > 0:
            option_list.action_first()

    @on(OptionList.OptionSelected, "#model_list")
    def _on_select(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(str(event.option.prompt))

    def action_move_up(self) -> None:
        option_list = self.query_one("#model_list", OptionList)
        if option_list.option_count > 0:
            option_list.action_cursor_up()

    def action_move_down(self) -> None:
        option_list = self.query_one("#model_list", OptionList)
        if option_list.option_count > 0:
            option_list.action_cursor_down()

    def action_accept(self) -> None:
        option_list = self.query_one("#model_list", OptionList)
        highlighted = option_list.highlighted_option
        if highlighted is not None:
            self.dismiss(str(highlighted.prompt))

    def action_cancel(self) -> None:
        self.dismiss(None)


class ChatBubble(Vertical):
    def __init__(
        self,
        role: str,
        *,
        model_name: str | None = None,
        markdown: str = "",
    ) -> None:
        classes = (
            "chat-bubble user-bubble"
            if role == "user"
            else "chat-bubble agent-bubble"
        )
        super().__init__(classes=classes)
        self._role = role
        self._model_name = model_name
        self._markdown = markdown
        self._tool_outputs: list[RenderableType] = []

    def compose(self) -> ComposeResult:
        yield Static(self._title_renderable(), classes="chat-title")
        yield Static("", classes="reasoning-line")
        yield Static("", classes="tool-status")
        yield Static("", classes="agent-status")
        yield Markdown(self._markdown, classes="chat-markdown")
        yield Static("", classes="tool-output")

    def _title_renderable(self) -> Text:
        if self._role == "user":
            return Text("User", style="bold")
        title = Text("Agent", style="bold")
        if self._model_name:
            title.append(" (", style="dim")
            title.append(self._model_name, style="grey62")
            title.append(")", style="dim")
        return title

    def markdown_widget(self) -> Markdown:
        return self.query_one(Markdown)

    async def set_markdown(self, markdown: str) -> None:
        await self.markdown_widget().update(markdown)

    def update_tool_status(self, text: str) -> None:
        self.query_one(".tool-status", Static).update(text)

    def clear_tool_status(self) -> None:
        self.query_one(".tool-status", Static).update("")

    def update_agent_status(self, text: str) -> None:
        self.query_one(".agent-status", Static).update(text)

    def clear_agent_status(self) -> None:
        self.query_one(".agent-status", Static).update("")

    def append_tool_output(self, renderable: RenderableType) -> None:
        self._tool_outputs.append(renderable)
        self.query_one(".tool-output", Static).update(Group(*self._tool_outputs))

    def update_reasoning_line(self, text: str) -> None:
        self.query_one(".reasoning-line", Static).update(text)

    def set_reasoning_summary(self, text: str) -> None:
        self.query_one(".reasoning-line", Static).update(text)


class ComposerInput(TextArea):
    BINDINGS = [
        Binding(
            "ctrl+backspace",
            "delete_word_left",
            "Delete word left",
            show=False,
        ),
    ]


class SubAgentCard(Vertical):
    _ACTIVE_STATUSES = {"running", "restarting", "terminating"}
    class Dismissed(Message):
        def __init__(self, subagent_id: str) -> None:
            super().__init__()
            self.subagent_id = subagent_id

    _STATUS_CLASSES = {
        "status-running",
        "status-restarting",
        "status-terminating",
        "status-completed",
        "status-failed",
        "status-terminated",
        "status-stuck",
    }
    _ACCENT_CLASSES = {
        "accent-a",
        "accent-b",
        "accent-c",
        "accent-d",
        "accent-e",
        "accent-f",
    }

    def __init__(self, subagent_id: str, snapshot: dict[str, object]) -> None:
        super().__init__(classes="subagent-card")
        self.subagent_id = subagent_id
        self._snapshot = dict(snapshot)

    def compose(self) -> ComposeResult:
        with Horizontal(classes="subagent-card-topbar"):
            yield Static("", classes="subagent-card-summary")
            yield Button("Close", classes="subagent-card-dismiss", variant="default")
        with Collapsible(
            title="Details",
            collapsed=True,
            classes="subagent-card-details-collapsible",
        ):
            yield Static("", classes="subagent-card-details")

    def on_mount(self) -> None:
        self.update_from_snapshot(self._snapshot)

    @on(Button.Pressed, ".subagent-card-dismiss")
    def _on_dismiss(self, _: Button.Pressed) -> None:
        self.post_message(self.Dismissed(self.subagent_id))

    def update_from_snapshot(self, snapshot: dict[str, object]) -> None:
        self._snapshot = dict(snapshot)
        status = str(snapshot.get("status", "unknown")).strip().lower() or "unknown"
        goal = str(snapshot.get("goal", "")).strip()
        current_activity = str(snapshot.get("current_activity", "")).strip()
        activity_detail = str(snapshot.get("activity_detail", "")).strip()
        latest_report = str(snapshot.get("latest_report", "")).strip()
        final_result = str(snapshot.get("final_result", "")).strip()
        error = str(snapshot.get("error", "")).strip()
        stop_reason = str(snapshot.get("stop_reason", "")).strip()
        tool_calls = int(snapshot.get("tool_calls", 0) or 0)
        output_chars = int(snapshot.get("output_chars", 0) or 0)
        attempt = int(snapshot.get("attempt", 1) or 1)
        short_id = self.subagent_id.removeprefix("subagent-")

        activity_text = current_activity or latest_report or status
        summary_goal = _single_line_preview(goal or "No goal provided.", 52)
        summary_activity = _single_line_preview(activity_text, 42)
        summary_text = (
            f"Agent {short_id}\n"
            f"Goal: {summary_goal}\n"
            f"Now: {summary_activity}"
        )
        self.query_one(".subagent-card-summary", Static).update(summary_text)

        details_lines = [
            f"ID: {self.subagent_id}",
            f"Status: {status}",
            f"Attempt: {attempt}",
            f"Goal: {goal or 'No goal provided.'}",
            f"Now: {activity_text or 'Working'}",
        ]
        if activity_detail:
            details_lines.append(f"Detail: {activity_detail}")
        details_lines.append(f"Tool calls: {tool_calls}")
        details_lines.append(f"Output chars: {output_chars}")
        if latest_report:
            details_lines.append(f"Latest report: {latest_report}")
        if final_result:
            details_lines.append(f"Final result: {final_result}")
        if error:
            details_lines.append(f"Error: {error}")
        if stop_reason:
            details_lines.append(f"Stop reason: {stop_reason}")
        self.query_one(".subagent-card-details", Static).update("\n".join(details_lines))

        card_title = f"{short_id} | {summary_activity or 'Working'}"
        self.query_one(Collapsible).title = card_title
        self.query_one(".subagent-card-dismiss", Button).disabled = (
            status in self._ACTIVE_STATUSES
        )

        self._apply_status_class(status)
        self._apply_accent_class()

    def _apply_status_class(self, status: str) -> None:
        for class_name in self._STATUS_CLASSES:
            self.remove_class(class_name)
        self.add_class(f"status-{status}")

    def _apply_accent_class(self) -> None:
        for class_name in self._ACCENT_CLASSES:
            self.remove_class(class_name)
        idx = sum(ord(ch) for ch in self.subagent_id) % 6
        self.add_class(("accent-a", "accent-b", "accent-c", "accent-d", "accent-e", "accent-f")[idx])
