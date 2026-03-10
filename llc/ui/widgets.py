from __future__ import annotations

from rich.console import Group, RenderableType
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Markdown, OptionList, Static, TextArea

from llc.models import AvailableModel


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
