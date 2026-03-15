from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

from textual import events, on, work
from textual.actions import SkipAction
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.worker import Worker, WorkerState, get_current_worker
from textual.widgets import Input, OptionList, Static

from llc.models import AvailableModel
from llc.service.engine import SessionEngine
from llc.service.events import (
    CommandOutput,
    ErrorOccurred,
    Event,
    SubagentStatusUpdate,
    TurnCompleted,
    TurnStarted,
)
from llc.ui.panels import (
    AgentsPanel,
    CommandSurfacePanel,
    ExecutionPanel,
    LogStreamPanel,
    MissionMessage,
    TopBarPanel,
)
from llc.ui.state import (
    LogCategory,
    MissionControlState,
    SessionPhase,
    UiMode,
)
from llc.ui.telemetry_mapper import TelemetryMapper
from llc.ui.theme import register_mission_control_themes

_DOUBLE_ESCAPE_WINDOW_S = 0.6
_AGENT_STATUS_POLL_INTERVAL_S = 0.8
_QUIET_NOTICE_TIMEOUT_S = 3.5
_AGENTS_MIN_WIDTH = 32
_AGENTS_MAX_WIDTH = 56
_AGENTS_DEFAULT_WIDTH = 42


class MissionControlApp(App[None]):
    CSS_PATH = "app.tcss"
    BINDINGS = [
        Binding("ctrl+enter,ctrl+j,ctrl+m", "send_message", "Send", show=False),
        Binding("ctrl+n,ctrl+o", "focus_command", "Command", show=False),
        Binding("ctrl+g", "toggle_agents_panel", "Agents", show=False),
        Binding("escape", "register_escape_interrupt", show=False, priority=True),
        Binding("ctrl+l", "toggle_log_focus", "Focus Logs", show=False),
        Binding("ctrl+a", "focus_agents", "Focus Agents", show=False),
        Binding("ctrl+e", "focus_execution", "Focus Execution", show=False),
        Binding("ctrl+u", "focus_overview", "Overview", show=False),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, engine: SessionEngine) -> None:
        super().__init__()
        self._engine = engine
        self._state = MissionControlState()
        self._mapper = TelemetryMapper(self._state)
        self._active_turn_worker: Worker[None] | None = None
        self._interrupt_in_progress = False
        self._last_escape_pressed_at = 0.0
        self._subagent_poll_task: asyncio.Task[None] | None = None
        self._message_widgets: dict[str, MissionMessage] = {}
        self._message_snapshots: dict[str, str] = {}
        self._rendered_log_count = 0
        self._rendered_tool_count = 0
        self._rendered_reasoning_snapshot: list[str] = []
        self._agents_resizing = False
        self._sync_lock = asyncio.Lock()

    def compose(self) -> ComposeResult:
        yield TopBarPanel(id="top_bar")
        with Horizontal(id="main_row"):
            yield ExecutionPanel(id="execution")
            yield Static("│", id="agents_resize_handle")
            yield AgentsPanel(id="agents")
        yield LogStreamPanel(id="log_panel")
        yield CommandSurfacePanel(id="command_surface")

    async def on_mount(self) -> None:
        register_mission_control_themes(self)
        self._mapper.set_model_name(self._engine.model_name)
        self._mapper.set_sub_agent_mode(self._engine.sub_agent_mode_enabled)
        await self._sync_view()
        self._agents().styles.width = _AGENTS_DEFAULT_WIDTH
        self._command_surface().focus_command()

        try:
            await self._engine.initialize()
        except Exception as exc:  # noqa: BLE001
            self._mapper.add_system_message(
                "Model initialization failed. Set MODEL_NAME and OPENAI_API_KEY. "
                f"Error: {exc!s}",
                category=LogCategory.ERR,
            )
            await self._sync_view()
            return

        self._refresh_runtime_model_cache()
        self._start_subagent_polling()
        self._mapper.add_system_message(
            "Connected. Enter sends. Use /model to switch model. Ctrl+Q quits.",
            category=LogCategory.SYS,
        )
        await self._sync_view()

    def on_unmount(self) -> None:
        task = self._subagent_poll_task
        if task is not None and not task.done():
            task.cancel()
        self._subagent_poll_task = None
        with contextlib.suppress(RuntimeError):
            asyncio.create_task(self._engine.shutdown())

    def _top_bar(self) -> TopBarPanel:
        return self.query_one("#top_bar", TopBarPanel)

    def _execution(self) -> ExecutionPanel:
        return self.query_one("#execution", ExecutionPanel)

    def _agents(self) -> AgentsPanel:
        return self.query_one("#agents", AgentsPanel)

    def _agents_resize_handle(self) -> Static:
        return self.query_one("#agents_resize_handle", Static)

    def _log_panel(self) -> LogStreamPanel:
        return self.query_one("#log_panel", LogStreamPanel)

    def _command_surface(self) -> CommandSurfacePanel:
        return self.query_one("#command_surface", CommandSurfacePanel)

    def _normalized_command(self, text: str) -> str:
        return " ".join(text.strip().split())

    def _quiet_command_kind(self, text: str) -> str | None:
        normalized = self._normalized_command(text)
        if normalized.startswith("/model "):
            return "model"
        if normalized == "/enable sub-agent-mode":
            return "enable_sub_agent_mode"
        return None

    def _notify_quiet_command(self, *, kind: str, message: str) -> None:
        compact = " ".join(message.replace("`", "").split())
        if not compact:
            return
        lower = compact.lower()
        severity = "information"
        if "failed" in lower or "error" in lower:
            severity = "error"
        elif lower.startswith("usage:"):
            severity = "warning"

        title = "Model" if kind == "model" else "Sub-agent mode"
        if kind == "model" and lower.startswith("now using "):
            compact = f"Switched to {compact[len('Now using '):].rstrip('.')}"
        if kind == "enable_sub_agent_mode" and lower.startswith("enabled sub-agent-mode"):
            compact = "Sub-agent mode enabled"

        self.notify(
            compact,
            title=title,
            severity=severity,
            timeout=_QUIET_NOTICE_TIMEOUT_S,
            markup=False,
        )

    def _consume_quiet_command_event(self, *, kind: str, event: Event) -> bool:
        if isinstance(event, CommandOutput):
            if event.message:
                self._notify_quiet_command(kind=kind, message=event.message)
            return True
        if isinstance(event, ErrorOccurred):
            self._notify_quiet_command(kind=kind, message=event.message)
            return True
        return isinstance(event, (TurnStarted, TurnCompleted))

    def _refresh_runtime_model_cache(self) -> None:
        available: list[AvailableModel] = self._engine.available_models
        self._mapper.set_available_models(available)
        self._mapper.set_model_name(self._engine.model_name)
        self._mapper.set_sub_agent_mode(self._engine.sub_agent_mode_enabled)

    def _set_busy(self, busy: bool) -> None:
        self._mapper.set_busy(busy)
        surface = self._command_surface()
        surface.set_busy(busy)
        if not busy and not surface.model_selector_visible():
            surface.focus_command()

    async def _sync_view(self) -> None:
        async with self._sync_lock:
            self._top_bar().render_state(self._state)
            self._sync_reasoning_timeline()
            self._sync_tool_timeline()
            await self._sync_messages()
            self._agents().render_workers(
                self._state.workers,
                self._state.top_bar.sub_agent_mode_enabled,
            )
            self._sync_logs()
            self._apply_mode()

    def _sync_tool_timeline(self) -> None:
        tool_count = len(self._state.tool_timeline)
        if tool_count < self._rendered_tool_count:
            self._execution().clear_tool_lines()
            self._rendered_tool_count = 0
        for entry in self._state.tool_timeline[self._rendered_tool_count :]:
            self._execution().append_tool_line(entry.timeline_line)
        self._rendered_tool_count = tool_count

    def _sync_reasoning_timeline(self) -> None:
        reasoning_lines = list(self._state.reasoning_lines)
        if reasoning_lines == self._rendered_reasoning_snapshot:
            return
        if (
            len(reasoning_lines) < len(self._rendered_reasoning_snapshot)
            or (
                len(reasoning_lines) == len(self._rendered_reasoning_snapshot)
                and reasoning_lines
                and reasoning_lines[-1] != self._rendered_reasoning_snapshot[-1]
            )
        ):
            self._execution().clear_reasoning_lines()
            for line in reasoning_lines:
                self._execution().append_reasoning_line(line)
            self._rendered_reasoning_snapshot = reasoning_lines
            return
        for line in reasoning_lines[len(self._rendered_reasoning_snapshot) :]:
            self._execution().append_reasoning_line(line)
        self._rendered_reasoning_snapshot = reasoning_lines

    async def _sync_messages(self) -> None:
        lane = self._execution().lane()
        for entry in self._state.messages:
            widget = self._message_widgets.get(entry.id)
            snapshot = entry.model_dump_json()
            if widget is None:
                widget = await lane.mount_entry(entry)
                self._message_widgets[entry.id] = widget
                self._message_snapshots[entry.id] = snapshot
                continue
            if self._message_snapshots.get(entry.id) == snapshot:
                continue
            await widget.sync_content(entry)
            self._message_snapshots[entry.id] = snapshot

    def _sync_logs(self) -> None:
        logs = list(self._state.logs)
        if len(logs) < self._rendered_log_count:
            self._log_panel().clear()
            self._rendered_log_count = 0
        for entry in logs[self._rendered_log_count :]:
            self._log_panel().append_entry(entry)
        self._rendered_log_count = len(logs)

    def _apply_mode(self) -> None:
        for class_name in ("mode-focus-logs", "mode-focus-agents", "mode-focus-execution"):
            self.remove_class(class_name)
        if self._state.mode == UiMode.FOCUS_LOGS:
            self.add_class("mode-focus-logs")
        elif self._state.mode == UiMode.FOCUS_AGENTS:
            self.add_class("mode-focus-agents")
        elif self._state.mode == UiMode.FOCUS_EXECUTION:
            self.add_class("mode-focus-execution")

    def action_send_message(self) -> None:
        if self.focused is not self._command_surface().command_input():
            raise SkipAction()
        self._submit_from_command_surface()

    def action_focus_command(self) -> None:
        self._command_surface().focus_command()

    def action_toggle_agents_panel(self) -> None:
        panel = self._agents()
        panel.display = not panel.display

    def action_toggle_log_focus(self) -> None:
        if self._state.mode == UiMode.FOCUS_LOGS:
            self._state.mode = UiMode.OVERVIEW
        else:
            self._state.mode = UiMode.FOCUS_LOGS
        self._apply_mode()

    def action_focus_agents(self) -> None:
        self._state.mode = UiMode.FOCUS_AGENTS
        self._apply_mode()

    def action_focus_execution(self) -> None:
        self._state.mode = UiMode.FOCUS_EXECUTION
        self._apply_mode()

    def action_focus_overview(self) -> None:
        self._state.mode = UiMode.OVERVIEW
        self._apply_mode()

    @on(events.MouseDown, "#agents_resize_handle")
    def _on_agents_resize_start(self, event: events.MouseDown) -> None:
        self._agents_resizing = True
        self._agents_resize_handle().capture_mouse(True)
        self.add_class("agents-resizing")
        self._resize_agents_panel(event.screen_x)
        event.stop()

    @on(events.MouseMove, "#agents_resize_handle")
    def _on_agents_resize_move(self, event: events.MouseMove) -> None:
        if not self._agents_resizing:
            return
        self._resize_agents_panel(event.screen_x)
        event.stop()

    @on(events.MouseUp, "#agents_resize_handle")
    def _on_agents_resize_end(self, _: events.MouseUp) -> None:
        self._stop_agents_resize()

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

    @on(Input.Submitted, "#command_input")
    def _on_command_submitted(self, _: Input.Submitted) -> None:
        self._submit_from_command_surface()

    @on(Input.Submitted, "#model_filter_input")
    def _on_model_filter_submitted(self, _: Input.Submitted) -> None:
        self._apply_model_from_selector()

    @on(Input.Changed, "#model_filter_input")
    def _on_model_filter_changed(self, event: Input.Changed) -> None:
        self._command_surface().update_model_filter(event.value)

    @on(OptionList.OptionSelected, "#model_option_list")
    def _on_model_option_selected(self, _: OptionList.OptionSelected) -> None:
        self._apply_model_from_selector()

    def on_key(self, event: events.Key) -> None:
        surface = self._command_surface()
        if not surface.model_selector_visible():
            return
        if self.focused is surface.model_filter_input():
            if event.key == "up":
                surface.move_model_cursor_up()
                event.stop()
                event.prevent_default()
                return
            if event.key == "down":
                surface.move_model_cursor_down()
                event.stop()
                event.prevent_default()
                return
        if event.key == "escape":
            surface.close_model_selector()
            event.stop()
            event.prevent_default()

    def on_mouse_up(self, _: events.MouseUp) -> None:
        if self._agents_resizing:
            self._stop_agents_resize()

    def _stop_agents_resize(self) -> None:
        self._agents_resizing = False
        self._agents_resize_handle().capture_mouse(False)
        self.remove_class("agents-resizing")

    def _resize_agents_panel(self, screen_x: int) -> None:
        new_width = max(self.size.width - screen_x - 1, _AGENTS_MIN_WIDTH)
        new_width = min(new_width, _AGENTS_MAX_WIDTH)
        self._agents().styles.width = new_width

    def _submit_from_command_surface(self) -> None:
        if self._state.busy:
            return
        surface = self._command_surface()
        raw = surface.command_value().strip()
        if not raw:
            return
        surface.clear_command()
        surface.focus_command()

        if raw == "/model":
            if not self._state.available_models:
                self.notify(
                    "No models available from provider.",
                    title="Model",
                    severity="warning",
                    timeout=_QUIET_NOTICE_TIMEOUT_S,
                    markup=False,
                )
                return
            surface.open_model_selector(self._state.available_models)
            return

        quiet_command = self._quiet_command_kind(raw)
        self._set_busy(True)
        self._start_user_turn(raw, quiet_command=quiet_command)

    def _apply_model_from_selector(self) -> None:
        if self._state.busy:
            return
        model_id = self._command_surface().highlighted_model_id()
        if not model_id:
            return
        self._command_surface().close_model_selector()
        self._set_busy(True)
        self._start_user_turn(f"/model {model_id}", quiet_command="model")

    def _start_user_turn(self, user_input: str, *, quiet_command: str | None = None) -> None:
        self._active_turn_worker = self._handle_user_turn(
            user_input,
            quiet_command=quiet_command,
        )

    @work(exclusive=False, exit_on_error=False)
    async def _handle_user_turn(
        self,
        user_input: str,
        *,
        quiet_command: str | None = None,
    ) -> None:
        worker = get_current_worker()
        try:
            if quiet_command is None:
                self._mapper.add_user_message(user_input)
                await self._sync_view()
            async for event in self._engine.send_message(user_input):
                if quiet_command is not None and self._consume_quiet_command_event(
                    kind=quiet_command,
                    event=event,
                ):
                    continue
                if isinstance(event, TurnCompleted) and event.compact_message:
                    self.notify(
                        "Context history compacted to keep runtime stable.",
                        title="Context",
                        severity="information",
                        timeout=_QUIET_NOTICE_TIMEOUT_S,
                        markup=False,
                    )
                self._mapper.process(event)
                await self._sync_view()
        finally:
            is_current_worker = self._active_turn_worker is worker
            if is_current_worker:
                self._active_turn_worker = None
            self._refresh_runtime_model_cache()
            if is_current_worker:
                self._set_busy(False)
                if self._state.top_bar.phase == SessionPhase.VERIFY:
                    self._state.top_bar.phase = SessionPhase.IDLE
            await self._sync_view()
            if is_current_worker and not self._command_surface().model_selector_visible():
                self._command_surface().focus_command()

    @work(exclusive=False, exit_on_error=False)
    async def _interrupt_session(self) -> None:
        self._set_busy(True)
        try:
            await self._cancel_active_turn_worker()
            self._state.active_message_id = None
            self._state.top_bar.phase = SessionPhase.IDLE
            async for event in self._engine.interrupt():
                self._mapper.process(event)
                await self._sync_view()
        finally:
            self._set_busy(False)
            if not self._command_surface().model_selector_visible():
                self._command_surface().focus_command()
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

    def _start_subagent_polling(self) -> None:
        task = self._subagent_poll_task
        if task is not None and not task.done():
            return
        self._subagent_poll_task = asyncio.create_task(self._poll_subagent_status())

    async def _poll_subagent_status(self) -> None:
        try:
            while True:
                report: dict[str, Any] = {"workers": []}
                try:
                    report = self._engine.get_subagent_report(include_all=True)
                except Exception:
                    report = {"workers": []}
                event = SubagentStatusUpdate(
                    workers=[
                        worker
                        for worker in report.get("workers", [])
                        if isinstance(worker, dict)
                    ],
                    active_count=int(report.get("active_count", 0) or 0),
                    max_sub_agents=int(report.get("max_sub_agents", 0) or 0),
                )
                self._mapper.process(event)
                await self._sync_view()
                await asyncio.sleep(_AGENT_STATUS_POLL_INTERVAL_S)
        except asyncio.CancelledError:
            return

