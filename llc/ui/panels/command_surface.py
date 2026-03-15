from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Input, OptionList, Static

from llc.ui.state import ModelOption


class CommandSurfacePanel(Vertical):
    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self._model_options: list[ModelOption] = []
        self._label_to_model_id: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        with Horizontal(id="command_row"):
            yield Static(">", id="command_prompt")
            yield Input(
                value="",
                placeholder="type instructions or /command",
                id="command_input",
            )
            yield Static("Enter send | Esc Esc interrupt | Ctrl+Q quit", id="command_hint")
        with Vertical(id="model_selector"):
            yield Static("MODEL", id="model_selector_title")
            yield Input(value="", placeholder="filter models", id="model_filter_input")
            yield OptionList(id="model_option_list")

    def on_mount(self) -> None:
        self.query_one("#model_selector", Vertical).display = False

    def command_input(self) -> Input:
        return self.query_one("#command_input", Input)

    def model_filter_input(self) -> Input:
        return self.query_one("#model_filter_input", Input)

    def model_option_list(self) -> OptionList:
        return self.query_one("#model_option_list", OptionList)

    def command_value(self) -> str:
        return self.command_input().value

    def clear_command(self) -> None:
        self.command_input().value = ""

    def focus_command(self) -> None:
        self.command_input().focus()

    def set_busy(self, busy: bool) -> None:
        self.command_input().read_only = busy

    def model_selector_visible(self) -> bool:
        return bool(self.query_one("#model_selector", Vertical).display)

    def open_model_selector(self, model_options: list[ModelOption]) -> None:
        self._model_options = list(model_options)
        panel = self.query_one("#model_selector", Vertical)
        panel.display = True
        self.model_filter_input().value = ""
        self._refresh_model_options("")
        self.model_filter_input().focus()

    def close_model_selector(self) -> None:
        self.query_one("#model_selector", Vertical).display = False
        self.model_filter_input().value = ""
        self.model_option_list().clear_options()
        self._label_to_model_id.clear()
        self.focus_command()

    def update_model_filter(self, query: str) -> None:
        self._refresh_model_options(query)

    def highlighted_model_id(self) -> str | None:
        highlighted = self.model_option_list().highlighted_option
        if highlighted is None:
            return None
        label = str(highlighted.prompt)
        return self._label_to_model_id.get(label)

    def move_model_cursor_up(self) -> None:
        if self.model_option_list().option_count <= 0:
            return
        self.model_option_list().action_cursor_up()

    def move_model_cursor_down(self) -> None:
        if self.model_option_list().option_count <= 0:
            return
        self.model_option_list().action_cursor_down()

    def _refresh_model_options(self, query: str) -> None:
        q = query.strip().lower()
        self.model_option_list().clear_options()
        self._label_to_model_id = {}
        options = self._model_options
        if q:
            options = [
                option
                for option in self._model_options
                if q in option.id.lower() or q in option.name.lower()
            ]
        for option in options[:200]:
            label = option.display_label
            self._label_to_model_id[label] = option.id
            self.model_option_list().add_option(label)
        if self.model_option_list().option_count > 0:
            self.model_option_list().action_first()
