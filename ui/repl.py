from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from rich.text import Text
from textual import on, work
from textual.actions import SkipAction
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Markdown, OptionList, Static, TextArea

from agent import build_agent_graph
from commands import CommandRegistry, ReplContext
from config import Settings
from hooks import AutoCompactHook, Hook, HookContext, TokenCounterHook
from models import AvailableModel, fetch_models
from ui.display import (
    collect_new_tool_calls,
    extract_reasoning,
    format_tool_args,
    message_text,
)

LLC_LOGO = r"""  ██╗     ██╗      ██████╗
  ██║     ██║     ██╔════╝
  ██║     ██║     ██║
  ██║     ██║     ██║
  ███████╗███████╗╚██████╗
  ╚══════╝╚══════╝ ╚═════╝"""

_REASONING_TOKENS_PER_LINE = 20
_REASONING_LINE_INTERVAL = 1.0
_DEBUG_LOG_PATH = Path("/home/tanay/Desktop/local-claude-code/.cursor/debug-eda7ca.log")
_DEBUG_LOG_FALLBACK_PATH = Path("/workspace/.cursor/debug-eda7ca.log")


# region debug token logs
def _debug_log(
    location: str,
    message: str,
    data: dict[str, Any],
    hypothesis_id: str,
    run_id: str,
) -> None:
    payload = {
        "sessionId": "eda7ca",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    for path in (_DEBUG_LOG_PATH, _DEBUG_LOG_FALLBACK_PATH):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, separators=(",", ":")) + "\n")
            break
        except Exception:
            continue


# endregion debug token logs


def _format_duration(seconds: float) -> str:
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


class _ReasoningTracker:
    def __init__(self) -> None:
        self.started_at: float | None = None
        self._tokens: list[str] = []
        self._last_emitted: float | None = None

    def feed(self, text: str) -> str | None:
        cleaned = text.strip()
        if not cleaned:
            return None
        if self.started_at is None:
            self.started_at = time.monotonic()
        words = cleaned.split()
        if not words:
            return None
        self._tokens.extend(words)
        now = time.monotonic()
        if len(self._tokens) < _REASONING_TOKENS_PER_LINE:
            return None
        if (
            self._last_emitted is not None
            and now - self._last_emitted < _REASONING_LINE_INTERVAL
        ):
            return None
        line_tokens = self._tokens[:_REASONING_TOKENS_PER_LINE]
        self._tokens = self._tokens[_REASONING_TOKENS_PER_LINE:]
        self._last_emitted = now
        return " ".join(line_tokens)

    @property
    def active(self) -> bool:
        return self.started_at is not None

    def finish(self) -> str:
        elapsed = 0.0
        if self.started_at is not None:
            elapsed = max(time.monotonic() - self.started_at, 0.0)
        self.started_at = None
        self._tokens.clear()
        self._last_emitted = None
        return f"  Reasoned for {_format_duration(elapsed)}"


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

    def compose(self) -> ComposeResult:
        yield Static(self._title_renderable(), classes="chat-title")
        yield Static("", classes="reasoning-line")
        yield Static("", classes="tool-status")
        yield Markdown(self._markdown, classes="chat-markdown")

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


class Repl(App[None]):
    CSS_PATH = "repl.tcss"
    BINDINGS = [
        Binding(
            "enter,return,ctrl+enter,ctrl+return,ctrl+j,ctrl+m",
            "send_message",
            "Send",
            priority=True,
        ),
        Binding(
            "shift+enter,shift+return,ctrl+n,ctrl+o,alt+enter,alt+return",
            "insert_newline",
            "New line",
            show=False,
            priority=True,
        ),
        Binding("ctrl+q", "quit", "Quit", priority=True),
    ]

    def __init__(self, settings: Settings, registry: CommandRegistry) -> None:
        super().__init__()
        self._settings = settings
        self._command_registry = registry
        self._ctx: ReplContext | None = None
        self._agent_thread_id = f"tui-{uuid.uuid4()}"
        self._token_hook = TokenCounterHook()
        self._hooks: list[Hook] = [self._token_hook, AutoCompactHook()]
        self._available_models: list[AvailableModel] = []
        self._turn_input = 0
        self._turn_output = 0
        self._turn_text_len = 0
        self._busy = False

    def compose(self) -> ComposeResult:
        yield Static(id="banner")
        with VerticalScroll(id="chat_scroll"):
            yield Vertical(id="chat_column")
        with Horizontal(id="composer"):
            yield ComposerInput(id="composer_input")
            yield Button("Send", id="send_button", variant="primary")

    async def on_mount(self) -> None:
        self._configure_layout()
        self._chat_scroll().anchor()
        self._composer_input().focus()
        self._configure_composer()
        self._refresh_banner()

        try:
            agent = build_agent_graph(self._settings)
        except Exception as exc:  # noqa: BLE001
            await self._append_system(
                "Model initialization failed. "
                "Set `MODEL_NAME`, `OPENAI_API_KEY`, and optionally "
                "`OPENAI_BASE_URL` in your environment or `.env` file.\n\n"
                f"Error: `{exc!s}`"
            )
            return

        self._ctx = ReplContext(
            settings=self._settings,
            agent=agent,
            thread_id=self._agent_thread_id,
        )
        # region debug token logs
        _debug_log(
            "ui/repl.py:on_mount",
            "repl mounted",
            {
                "model_name": self._settings.model_name,
                "has_api_key": bool(self._settings.openai_api_key),
                "base_url": bool(self._settings.openai_base_url),
            },
            "H5",
            self._agent_thread_id,
        )
        # endregion debug token logs
        await self._append_system(
            "Connected. Type `/help` for commands. "
            "`Ctrl+Enter` to send (`Enter` on many terminals). "
            "`Ctrl+N`/`Ctrl+O` insert newline. `Ctrl+Q` to quit."
        )
        self._load_available_models()

    def _configure_composer(self) -> None:
        composer = self._composer_input()
        composer.show_line_numbers = False
        composer.soft_wrap = True

    def _configure_layout(self) -> None:
        self.query_one("#banner", Static).styles.dock = "top"
        self.query_one("#composer", Horizontal).styles.dock = "bottom"
        self._chat_scroll().styles.height = "1fr"

    def _chat_scroll(self) -> VerticalScroll:
        return self.query_one("#chat_scroll", VerticalScroll)

    def _chat_column(self) -> Vertical:
        return self.query_one("#chat_column", Vertical)

    def _composer_input(self) -> ComposerInput:
        return self.query_one("#composer_input", ComposerInput)

    def _set_send_enabled(self, enabled: bool) -> None:
        self.query_one("#send_button", Button).disabled = not enabled

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._set_send_enabled(not busy)

    def _current_model_name(self) -> str:
        if self._ctx is not None:
            return self._ctx.settings.model_name
        return self._settings.model_name

    def _refresh_banner(self) -> None:
        model_name = self._current_model_name()
        inp = self._token_hook.session_input
        out = self._token_hook.session_output
        # region debug token logs
        _debug_log(
            "ui/repl.py:_refresh_banner",
            "banner refreshed",
            {"model_name": model_name, "session_input": inp, "session_output": out},
            "H4",
            self._agent_thread_id,
        )
        # endregion debug token logs
        banner = Text()
        banner.append(LLC_LOGO, style="bold cyan")
        banner.append("\n  Model: ", style="dim")
        banner.append(model_name, style="bold")
        banner.append("   |   ", style="dim")
        banner.append(f"Tokens: {inp:,} in / {out:,} out", style="dim")
        if self._token_hook.session_cost > 0:
            banner.append(f"   |   ${self._token_hook.session_cost:.4f}", style="dim")
        banner.append("\n  Type ", style="dim")
        banner.append("/help", style="bold")
        banner.append(" for commands. ", style="dim")
        banner.append("/model", style="bold")
        banner.append(" to switch models.", style="dim")
        self.query_one("#banner", Static).update(banner)

    async def _append_message(
        self,
        role: str,
        markdown: str,
        *,
        model_name: str | None = None,
    ) -> ChatBubble:
        bubble = ChatBubble(role, markdown=markdown, model_name=model_name)
        await self._chat_column().mount(bubble)
        self._chat_scroll().scroll_end(animate=False)
        return bubble

    async def _append_system(self, text: str) -> None:
        label = Static(text, classes="system-message")
        await self._chat_column().mount(label)
        self._chat_scroll().scroll_end(animate=False)

    @on(Button.Pressed, "#send_button")
    def _on_send_button(self, _: Button.Pressed) -> None:
        self._do_send()

    def _composer_is_focused(self) -> bool:
        focused = self.focused
        return focused is self._composer_input()

    def action_insert_newline(self) -> None:
        if not self._composer_is_focused():
            raise SkipAction()
        composer = self._composer_input()
        composer.insert("\n")
        composer.focus()

    def action_send_message(self) -> None:
        if not self._composer_is_focused():
            raise SkipAction()
        self._do_send()

    def _do_send(self) -> None:
        if self._busy or self._ctx is None:
            return
        raw = self._composer_input().text.strip()
        if not raw:
            return
        self._composer_input().clear()
        self._composer_input().focus()

        if raw == "/model" or raw.startswith("/model "):
            arg = raw[6:].strip()
            if not arg and self._available_models:
                self._show_model_picker()
                return
        self._set_busy(True)
        self._handle_user_turn(raw)

    def _show_model_picker(self) -> None:
        if not self._available_models:
            return

        def _on_result(selected: str | None) -> None:
            if selected is None:
                self._composer_input().focus()
                return
            self._set_busy(True)
            self._handle_user_turn(f"/model {selected}")

        self.push_screen(
            ModelPickerScreen(self._available_models), callback=_on_result
        )

    @work(exclusive=False, exit_on_error=False)
    async def _handle_user_turn(self, user_input: str) -> None:
        try:
            await self._append_message("user", user_input)
            if self._ctx is None:
                return

            match = self._command_registry.match(user_input)
            if match:
                command, args = match
                # region debug token logs
                _debug_log(
                    "ui/repl.py:_handle_user_turn",
                    "command matched",
                    {"command": command.name, "args_len": len(args)},
                    "H5",
                    self._agent_thread_id,
                )
                # endregion debug token logs
                await self._run_command(command, args)
                return

            # region debug token logs
            _debug_log(
                "ui/repl.py:_handle_user_turn",
                "agent turn matched",
                {"user_len": len(user_input)},
                "H5",
                self._agent_thread_id,
            )
            # endregion debug token logs
            bubble = await self._append_message(
                "agent",
                "",
                model_name=self._ctx.settings.model_name,
            )
            await self._stream_agent_response(user_input, bubble)
        finally:
            self._set_busy(False)

    async def _run_command(self, command: Any, args: str) -> None:
        if self._ctx is None:
            return
        result = await command.execute(args, self._ctx)
        if result.message:
            await self._append_message(
                "agent",
                result.message,
                model_name=self._ctx.settings.model_name,
            )
        self._refresh_banner()
        if result.should_exit:
            self.exit()

    async def _stream_agent_response(
        self, user_input: str, bubble: ChatBubble
    ) -> None:
        if self._ctx is None:
            return

        # region debug token logs
        _debug_log(
            "ui/repl.py:_stream_agent_response",
            "turn started",
            {"user_len": len(user_input), "model_name": self._current_model_name()},
            "H5",
            self._agent_thread_id,
        )
        # endregion debug token logs
        md_stream = Markdown.get_stream(bubble.markdown_widget())
        fallback_text = ""
        saw_text = False
        seen_ids: set[str] = set()
        reasoning = _ReasoningTracker()
        tool_idx = 0

        try:
            async for mode, chunk in self._ctx.agent.astream(
                {"messages": [HumanMessage(content=user_input)]},
                config={"configurable": {"thread_id": self._agent_thread_id}},
                stream_mode=["messages", "updates"],
            ):
                if mode == "messages":
                    msg_chunk, meta = chunk
                    if meta.get("langgraph_node") != "llm":
                        continue
                    self._extract_usage(msg_chunk)

                    r_text = extract_reasoning(msg_chunk)
                    if r_text:
                        line = reasoning.feed(r_text)
                        if line is not None:
                            bubble.update_reasoning_line(f"  {line}")
                        continue

                    text = message_text(msg_chunk.content)
                    if text:
                        if reasoning.active:
                            bubble.set_reasoning_summary(reasoning.finish())
                        saw_text = True
                        self._turn_text_len += len(text)
                        await md_stream.write(text)

                if mode != "updates" or not isinstance(chunk, dict):
                    continue

                for node_update in chunk.values():
                    if not isinstance(node_update, dict):
                        continue
                    for msg in node_update.get("messages", []):
                        if not isinstance(msg, AIMessage):
                            continue
                        self._extract_usage(msg)
                        if getattr(msg, "tool_calls", None):
                            continue
                        t = message_text(msg.content)
                        if t:
                            fallback_text = t

                new_tools = collect_new_tool_calls(chunk, seen_ids)
                for tc in new_tools:
                    tool_idx += 1
                    name = tc.get("name", "tool")
                    args = tc.get("args", {})
                    args_str = format_tool_args(args)
                    line = f"  ↳ [{tool_idx}] {name}"
                    if args_str:
                        line += f"({args_str})"
                    bubble.update_tool_status(line)

            await md_stream.stop()

            if reasoning.active:
                bubble.set_reasoning_summary(reasoning.finish())

            if not saw_text and fallback_text.strip():
                await bubble.set_markdown(fallback_text)
        except asyncio.CancelledError:
            await md_stream.stop()
            raise
        except Exception as exc:  # noqa: BLE001
            # region debug token logs
            _debug_log(
                "ui/repl.py:_stream_agent_response",
                "stream failed",
                {"error": str(exc)},
                "H5",
                self._agent_thread_id,
            )
            # endregion debug token logs
            await md_stream.stop()
            await bubble.set_markdown(f"Request failed: `{exc!s}`")
        finally:
            bubble.clear_tool_status()
            await self._finish_turn()

    async def _finish_turn(self) -> None:
        turn_input = self._turn_input
        turn_output = self._turn_output
        text_len = self._turn_text_len
        # region debug token logs
        _debug_log(
            "ui/repl.py:_finish_turn",
            "finish turn entry",
            {
                "turn_input": turn_input,
                "turn_output": turn_output,
                "text_len": text_len,
                "has_ctx": self._ctx is not None,
            },
            "H2",
            self._agent_thread_id,
        )
        # endregion debug token logs
        self._turn_input = 0
        self._turn_output = 0
        self._turn_text_len = 0

        try:
            if turn_input == 0 and turn_output == 0 and text_len > 0:
                turn_output = max(1, text_len // 4)
                # region debug token logs
                _debug_log(
                    "ui/repl.py:_finish_turn",
                    "fallback output estimate used",
                    {"estimated_output": turn_output, "text_len": text_len},
                    "H2",
                    self._agent_thread_id,
                )
                # endregion debug token logs

            # region debug token logs
            raw_hooks = getattr(self, "_hooks", None)
            _debug_log(
                "ui/repl.py:_finish_turn",
                "before hook loop",
                {
                    "internal_thread_id_type": type(
                        getattr(self, "_thread_id", None)
                    ).__name__,
                    "agent_thread_id_type": type(self._agent_thread_id).__name__,
                    "ctx_thread_id_type": (
                        type(self._ctx.thread_id).__name__ if self._ctx is not None else None
                    ),
                    "hooks_type": type(raw_hooks).__name__,
                    "hooks_len": len(raw_hooks) if isinstance(raw_hooks, list) else None,
                    "hooks_items": (
                        [type(h).__name__ for h in raw_hooks]
                        if isinstance(raw_hooks, list)
                        else None
                    ),
                },
                "H6",
                self._agent_thread_id,
            )
            # endregion debug token logs

            if self._ctx is not None:
                hook_ctx = HookContext(
                    agent=self._ctx.agent,
                    thread_id=self._ctx.thread_id,
                    settings=self._ctx.settings,
                    last_turn_input_tokens=turn_input,
                    last_turn_output_tokens=turn_output,
                    available_models=self._available_models,
                    compact_prompt=self._ctx.settings.compact_prompt,
                )
                # region debug token logs
                _debug_log(
                    "ui/repl.py:_finish_turn",
                    "hook context created",
                    {
                        "turn_input": turn_input,
                        "turn_output": turn_output,
                        "model_count": len(self._available_models),
                    },
                    "H6",
                    self._agent_thread_id,
                )
                # endregion debug token logs
                for hook in self._hooks:
                    try:
                        message = await hook.after_turn(hook_ctx)
                    except Exception as exc:  # noqa: BLE001
                        message = f"Hook failed: `{exc!s}`"
                    if message:
                        await self._append_system(message)
            self._refresh_banner()
        except Exception as exc:
            # region debug token logs
            _debug_log(
                "ui/repl.py:_finish_turn",
                "finish turn failed",
                {"error": str(exc), "error_type": type(exc).__name__},
                "H6",
                self._agent_thread_id,
            )
            # endregion debug token logs
            raise

    def _extract_usage(self, msg: Any) -> None:
        usage = getattr(msg, "usage_metadata", None)
        if not usage:
            return
        before_input = self._turn_input
        before_output = self._turn_output
        if isinstance(usage, dict):
            self._turn_input += usage.get("input_tokens", 0) or 0
            self._turn_output += usage.get("output_tokens", 0) or 0
        else:
            self._turn_input += getattr(usage, "input_tokens", 0) or 0
            self._turn_output += getattr(usage, "output_tokens", 0) or 0
        # region debug token logs
        _debug_log(
            "ui/repl.py:_extract_usage",
            "usage extracted",
            {
                "msg_type": type(msg).__name__,
                "before_input": before_input,
                "before_output": before_output,
                "after_input": self._turn_input,
                "after_output": self._turn_output,
                "usage_type": type(usage).__name__,
            },
            "H1",
            self._agent_thread_id,
        )
        # endregion debug token logs

    @work(exclusive=True, exit_on_error=False)
    async def _load_available_models(self) -> None:
        models = await fetch_models(
            base_url=self._settings.openai_base_url,
            api_key=self._settings.openai_api_key,
        )
        self._available_models = models
        self._refresh_banner()
