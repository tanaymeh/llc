from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

from rich.text import Text
from textual import on, work
from textual.actions import SkipAction
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.worker import Worker, WorkerState, get_current_worker
from textual.widgets import Button, Markdown, Static

from llc.models import AvailableModel
from llc.service.engine import SessionEngine
from llc.service.events import (
    CommandOutput,
    ErrorOccurred,
    ReasoningDelta,
    SubagentStatusUpdate,
    TextDelta,
    ToolCallStarted,
    ToolResultEvent,
    TurnCompleted,
    UsageUpdate,
)
from llc.ui.display import format_tool_args
from llc.ui.rendering import format_duration, render_user_facing_tool_output
from llc.ui.widgets import ChatBubble, ComposerInput, ModelPickerScreen, SubAgentCard

LLC_LOGO = r"""  ██╗     ██╗      ██████╗
  ██║     ██║     ██╔════╝
  ██║     ██║     ██║
  ██║     ██║     ██║
  ███████╗███████╗╚██████╗
  ╚══════╝╚══════╝ ╚═════╝"""

_REASONING_TOKENS_PER_LINE = 20
_REASONING_LINE_INTERVAL = 1.0
_AGENT_STATUS_POLL_INTERVAL_S = 0.8
_MAX_AGENT_STATUS_LINES = 5
_WORKER_TASK_PREVIEW_CHARS = 56
_ACTIVE_WORKER_STATUSES = {"running", "restarting", "terminating"}
_DOUBLE_ESCAPE_WINDOW_S = 0.6
_SUBAGENT_PANEL_EMPTY_ACTIVE = "No sub-agents are currently running."
_SUBAGENT_PANEL_EMPTY_PAST = "No past sub-agents yet."


def _single_line_preview(text: str, limit: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def _format_agent_status(report: dict[str, Any]) -> str:
    workers = report.get("workers", [])
    if not isinstance(workers, list):
        return ""

    known_workers = [worker for worker in workers if isinstance(worker, dict)]
    active_workers = [
        worker
        for worker in known_workers
        if str(worker.get("status", "")) in _ACTIVE_WORKER_STATUSES
    ]
    if not active_workers:
        return ""

    inactive_count = max(len(known_workers) - len(active_workers), 0)
    lines: list[str] = [
        f"  ↳ Workers: {len(active_workers)} active, {inactive_count} done/stopped"
    ]
    for index, worker in enumerate(active_workers[:_MAX_AGENT_STATUS_LINES], start=1):
        raw_id = str(worker.get("id", f"agent-{index}"))
        short_id = raw_id.removeprefix("subagent-")
        status = str(worker.get("status", "running"))
        task = _single_line_preview(
            str(worker.get("task", "")),
            _WORKER_TASK_PREVIEW_CHARS,
        )
        if not task:
            task = _single_line_preview(
                str(worker.get("latest_report", "working")),
                _WORKER_TASK_PREVIEW_CHARS,
            )
        lines.append(f"  ↳ Agent #{index} ({short_id}) {status}: {task}")

    remainder = len(active_workers) - _MAX_AGENT_STATUS_LINES
    if remainder > 0:
        lines.append(f"  ↳ +{remainder} more active agents")
    return "\n".join(lines)


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
        return f"  Reasoned for {format_duration(elapsed)}"


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
        Binding("ctrl+g", "toggle_agents_panel", "Agents", priority=True),
        Binding("escape", "register_escape_interrupt", show=False, priority=True),
        Binding("ctrl+q", "quit", "Quit", priority=True),
    ]

    def __init__(self, engine: SessionEngine) -> None:
        super().__init__()
        self._engine = engine
        self._available_models: list[AvailableModel] = []
        self._usage = UsageUpdate()
        self._busy = False
        self._active_turn_worker: Worker[None] | None = None
        self._last_escape_pressed_at: float = 0.0
        self._interrupt_in_progress = False
        self._subagent_cards: dict[str, SubAgentCard] = {}
        self._dismissed_subagent_ids: set[str] = set()
        self._agents_panel_visible = False
        self._subagent_panel_task: asyncio.Task[None] | None = None

    def compose(self) -> ComposeResult:
        yield Static(id="banner")
        with Vertical(id="agents_panel"):
            yield Static("Sub-Agents", id="agents_panel_title")
            with VerticalScroll(id="agents_panel_scroll"):
                with Vertical(id="agents_panel_column"):
                    yield Static("Active Sub-Agents", classes="agents-section-title")
                    yield Static(_SUBAGENT_PANEL_EMPTY_ACTIVE, id="agents_active_empty")
                    yield Vertical(id="agents_active_column")
                    yield Static("Past Sub-Agents", classes="agents-section-title")
                    yield Static(_SUBAGENT_PANEL_EMPTY_PAST, id="agents_past_empty")
                    yield Vertical(id="agents_past_column")
        with VerticalScroll(id="chat_scroll"):
            yield Vertical(id="chat_column")
        with Horizontal(id="composer"):
            yield ComposerInput(id="composer_input")
            yield Button("Agents", id="agents_button")
            yield Button("Send", id="send_button", variant="primary")

    async def on_mount(self) -> None:
        self._configure_layout()
        self._chat_scroll().anchor()
        self._composer_input().focus()
        self._configure_composer()
        self._set_agents_panel_visible(False)
        self._refresh_banner()

        try:
            await self._engine.initialize()
        except Exception as exc:  # noqa: BLE001
            await self._append_system(
                "Model initialization failed. "
                "Set `MODEL_NAME`, `OPENAI_API_KEY`, and optionally "
                "`OPENAI_BASE_URL` in your environment or `.env` file.\n\n"
                f"Error: `{exc!s}`"
            )
            return

        self._available_models = self._engine.available_models
        self._start_subagent_panel_polling()
        await self._append_system(
            "Connected. Type `/help` for commands. "
            "`Ctrl+Enter` to send (`Enter` on many terminals). "
            "`Ctrl+N`/`Ctrl+O` insert newline. `Ctrl+Q` to quit."
        )
        self._refresh_banner()

    def on_unmount(self) -> None:
        task = self._subagent_panel_task
        if task is not None and not task.done():
            task.cancel()
        self._subagent_panel_task = None
        with contextlib.suppress(RuntimeError):
            asyncio.create_task(self._engine.shutdown())

    def _configure_composer(self) -> None:
        composer = self._composer_input()
        composer.show_line_numbers = False
        composer.soft_wrap = True

    def _configure_layout(self) -> None:
        self.query_one("#banner", Static).styles.dock = "top"
        self.query_one("#agents_panel", Vertical).styles.dock = "right"
        self.query_one("#composer", Horizontal).styles.dock = "bottom"
        self._chat_scroll().styles.height = "1fr"

    def _chat_scroll(self) -> VerticalScroll:
        return self.query_one("#chat_scroll", VerticalScroll)

    def _chat_column(self) -> Vertical:
        return self.query_one("#chat_column", Vertical)

    def _composer_input(self) -> ComposerInput:
        return self.query_one("#composer_input", ComposerInput)

    def _agents_panel(self) -> Vertical:
        return self.query_one("#agents_panel", Vertical)

    def _agents_active_column(self) -> Vertical:
        return self.query_one("#agents_active_column", Vertical)

    def _agents_past_column(self) -> Vertical:
        return self.query_one("#agents_past_column", Vertical)

    def _agents_active_empty(self) -> Static:
        return self.query_one("#agents_active_empty", Static)

    def _agents_past_empty(self) -> Static:
        return self.query_one("#agents_past_empty", Static)

    def _set_send_enabled(self, enabled: bool) -> None:
        self.query_one("#send_button", Button).disabled = not enabled

    def _set_agents_panel_visible(self, visible: bool) -> None:
        self._agents_panel_visible = visible
        panel = self._agents_panel()
        panel.display = visible
        button = self.query_one("#agents_button", Button)
        button.variant = "primary" if visible else "default"

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._set_send_enabled(not busy)

    def _refresh_banner(self) -> None:
        banner = Text()
        banner.append(LLC_LOGO, style="bold cyan")
        banner.append("\n  Model: ", style="dim")
        banner.append(self._engine.model_name, style="bold")
        banner.append("   |   ", style="dim")
        banner.append(
            f"Tokens: {self._usage.session_input_tokens:,} in / {self._usage.session_output_tokens:,} out",
            style="dim",
        )
        if self._usage.session_cost > 0:
            banner.append(f"   |   ${self._usage.session_cost:.4f}", style="dim")
        banner.append("\n  Type ", style="dim")
        banner.append("/help", style="bold")
        banner.append(" for commands. ", style="dim")
        banner.append("/model", style="bold")
        banner.append(" to switch models.", style="dim")
        if self._engine.sub_agent_mode_enabled:
            banner.append("  ", style="dim")
            banner.append("sub-agent mode: ", style="dim")
            banner.append("ON", style="bold green")
        self.query_one("#banner", Static).update(banner)

    def _start_subagent_panel_polling(self) -> None:
        task = self._subagent_panel_task
        if task is not None and not task.done():
            return
        self._subagent_panel_task = asyncio.create_task(self._poll_subagent_panel())

    async def _poll_subagent_panel(self) -> None:
        try:
            while True:
                report: dict[str, Any] = {"workers": []}
                try:
                    report = self._engine.get_subagent_report(include_all=True)
                except Exception:
                    report = {"workers": []}
                await self._sync_subagent_panel(report)
                await asyncio.sleep(_AGENT_STATUS_POLL_INTERVAL_S)
        except asyncio.CancelledError:
            return

    async def _sync_subagent_panel(self, report: dict[str, Any]) -> None:
        workers_raw = report.get("workers", [])
        workers = [worker for worker in workers_raw if isinstance(worker, dict)]
        seen_ids: set[str] = set()
        active_count = 0
        past_count = 0

        for worker in workers:
            subagent_id = str(worker.get("id", "")).strip()
            if not subagent_id:
                continue
            if subagent_id in self._dismissed_subagent_ids:
                continue
            seen_ids.add(subagent_id)
            status = str(worker.get("status", "")).strip()
            is_active = status in _ACTIVE_WORKER_STATUSES
            target_column = (
                self._agents_active_column() if is_active else self._agents_past_column()
            )
            if is_active:
                active_count += 1
            else:
                past_count += 1

            card = self._subagent_cards.get(subagent_id)
            if card is None:
                card = SubAgentCard(subagent_id, worker)
                self._subagent_cards[subagent_id] = card
                await target_column.mount(card)
                continue

            card.update_from_snapshot(worker)
            if card.parent is not target_column:
                if card.parent is not None:
                    await card.remove()
                await target_column.mount(card)

        stale_ids = [
            subagent_id
            for subagent_id in self._subagent_cards
            if subagent_id not in seen_ids
        ]
        for subagent_id in stale_ids:
            card = self._subagent_cards.pop(subagent_id)
            if card.parent is not None:
                await card.remove()

        self._agents_active_empty().display = active_count == 0
        self._agents_past_empty().display = past_count == 0

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

    @on(Button.Pressed, "#agents_button")
    def _on_agents_button(self, _: Button.Pressed) -> None:
        self.action_toggle_agents_panel()

    @on(SubAgentCard.Dismissed)
    async def _on_subagent_card_dismissed(
        self,
        event: SubAgentCard.Dismissed,
    ) -> None:
        subagent_id = event.subagent_id
        self._dismissed_subagent_ids.add(subagent_id)
        card = self._subagent_cards.pop(subagent_id, None)
        if card is not None and card.parent is not None:
            await card.remove()
        self._agents_active_empty().display = len(self._agents_active_column().children) == 0
        self._agents_past_empty().display = len(self._agents_past_column().children) == 0

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

    def action_toggle_agents_panel(self) -> None:
        self._set_agents_panel_visible(not self._agents_panel_visible)

    def action_register_escape_interrupt(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_escape_pressed_at
        self._last_escape_pressed_at = now
        if elapsed > _DOUBLE_ESCAPE_WINDOW_S:
            return
        self._last_escape_pressed_at = 0.0
        self._start_interrupt()

    def _start_interrupt(self) -> None:
        if self._interrupt_in_progress:
            return
        self._interrupt_in_progress = True
        self._interrupt_session()

    def _start_user_turn(self, user_input: str) -> None:
        self._active_turn_worker = self._handle_user_turn(user_input)

    def _do_send(self) -> None:
        if self._busy:
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
        self._start_user_turn(raw)

    def _show_model_picker(self) -> None:
        if not self._available_models:
            return

        def _on_result(selected: str | None) -> None:
            if selected is None:
                self._composer_input().focus()
                return
            self._set_busy(True)
            self._start_user_turn(f"/model {selected}")

        self.push_screen(
            ModelPickerScreen(self._available_models), callback=_on_result
        )

    @work(exclusive=False, exit_on_error=False)
    async def _handle_user_turn(self, user_input: str) -> None:
        worker = get_current_worker()
        try:
            await self._append_message("user", user_input)
            await self._stream_engine_response(user_input)
            self._available_models = self._engine.available_models
        finally:
            if self._active_turn_worker is worker:
                self._active_turn_worker = None
            self._set_busy(False)

    @work(exclusive=False, exit_on_error=False)
    async def _interrupt_session(self) -> None:
        try:
            await self._cancel_active_turn_worker()
            self._set_busy(False)
            self._composer_input().focus()
            async for event in self._engine.interrupt():
                if isinstance(event, CommandOutput) and event.message:
                    await self._append_message(
                        "agent",
                        event.message,
                        model_name=self._engine.model_name,
                    )
                elif isinstance(event, ErrorOccurred):
                    await self._append_system(event.message)
        finally:
            self._interrupt_in_progress = False

    async def _cancel_active_turn_worker(self) -> None:
        worker = self._active_turn_worker
        if worker is None:
            return
        if worker.state in (WorkerState.PENDING, WorkerState.RUNNING):
            worker.cancel()
            with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError, Exception):
                await asyncio.wait_for(worker.wait(), timeout=1.5)
        self._active_turn_worker = None

    async def _stream_engine_response(self, user_input: str) -> None:
        stream_bubble: ChatBubble | None = None
        md_stream: Any | None = None
        reasoning = _ReasoningTracker()
        agent_status_stop = asyncio.Event()
        agent_status_task: asyncio.Task[None] | None = None

        async def ensure_stream_bubble() -> ChatBubble:
            nonlocal stream_bubble, md_stream, agent_status_task
            if stream_bubble is None:
                stream_bubble = await self._append_message(
                    "agent",
                    "",
                    model_name=self._engine.model_name,
                )
                md_stream = Markdown.get_stream(stream_bubble.markdown_widget())
                if self._engine.sub_agent_mode_enabled:
                    agent_status_task = asyncio.create_task(
                        self._poll_agent_status(stream_bubble, agent_status_stop)
                    )
            return stream_bubble

        try:
            async for event in self._engine.send_message(user_input):
                if isinstance(event, TextDelta):
                    bubble = await ensure_stream_bubble()
                    if reasoning.active:
                        bubble.set_reasoning_summary(reasoning.finish())
                    if md_stream is not None:
                        await md_stream.write(event.text)
                    continue

                if isinstance(event, ReasoningDelta):
                    bubble = await ensure_stream_bubble()
                    line = reasoning.feed(event.text)
                    if line is not None:
                        bubble.update_reasoning_line(f"  {line}")
                    continue

                if isinstance(event, ToolCallStarted):
                    bubble = await ensure_stream_bubble()
                    args_str = format_tool_args(event.args)
                    line = f"  ↳ [{event.tool_index}] {event.tool_name}"
                    if args_str:
                        line += f"({args_str})"
                    bubble.update_tool_status(line)
                    continue

                if isinstance(event, ToolResultEvent):
                    if not event.user_facing:
                        continue
                    bubble = await ensure_stream_bubble()
                    bubble.append_tool_output(
                        render_user_facing_tool_output(
                            event.tool_name,
                            event.render_mode,
                            event.content,
                        )
                    )
                    continue

                if isinstance(event, SubagentStatusUpdate):
                    report = {
                        "workers": event.workers,
                        "active_count": event.active_count,
                        "max_sub_agents": event.max_sub_agents,
                    }
                    if stream_bubble is not None:
                        stream_bubble.update_agent_status(_format_agent_status(report))
                    await self._sync_subagent_panel(report)
                    continue

                if isinstance(event, UsageUpdate):
                    self._usage = event
                    self._refresh_banner()
                    continue

                if isinstance(event, CommandOutput):
                    if event.message:
                        await self._append_message(
                            "agent",
                            event.message,
                            model_name=self._engine.model_name,
                        )
                    if event.should_exit:
                        self.exit()
                    continue

                if isinstance(event, ErrorOccurred):
                    if stream_bubble is None:
                        await self._append_message(
                            "agent",
                            f"Request failed: `{event.message}`",
                            model_name=self._engine.model_name,
                        )
                    elif md_stream is not None:
                        await md_stream.stop()
                        md_stream = None
                        await stream_bubble.set_markdown(f"Request failed: `{event.message}`")
                    else:
                        await stream_bubble.set_markdown(f"Request failed: `{event.message}`")
                    continue

                if isinstance(event, TurnCompleted):
                    if reasoning.active and stream_bubble is not None:
                        stream_bubble.set_reasoning_summary(reasoning.finish())
                    if event.compact_message:
                        await self._append_system(event.compact_message)
        finally:
            agent_status_stop.set()
            if agent_status_task is not None:
                with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                    await asyncio.wait_for(agent_status_task, timeout=1.5)
                if not agent_status_task.done():
                    agent_status_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await agent_status_task
            if md_stream is not None:
                with contextlib.suppress(Exception):
                    await md_stream.stop()
            if stream_bubble is not None:
                stream_bubble.clear_tool_status()
                stream_bubble.clear_agent_status()
            self._refresh_banner()

    async def _poll_agent_status(
        self,
        bubble: ChatBubble,
        stop_signal: asyncio.Event,
    ) -> None:
        while not stop_signal.is_set():
            status_text = ""
            try:
                report = self._engine.get_subagent_report()
                status_text = _format_agent_status(report)
            except Exception:
                status_text = ""
            bubble.update_agent_status(status_text)
            try:
                await asyncio.wait_for(
                    stop_signal.wait(),
                    timeout=_AGENT_STATUS_POLL_INTERVAL_S,
                )
            except asyncio.TimeoutError:
                continue

