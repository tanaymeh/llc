from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from rich.text import Text
from textual import on, work
from textual.actions import SkipAction
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Markdown, Static

from llc.agent import build_agent_graph
from llc.commands import CommandRegistry, ReplContext
from llc.config import Settings
from llc.agent.hooks import AutoCompactHook, Hook, HookContext, TokenCounterHook
from llc.agent.subagents import SubAgentRuntime
from llc.models import AvailableModel, fetch_models
from llc.ui.display import (
    collect_new_user_facing_tool_results,
    collect_new_tool_calls,
    extract_reasoning,
    format_tool_args,
    message_text,
)
from llc.ui.rendering import format_duration, render_user_facing_tool_output
from llc.ui.widgets import ChatBubble, ComposerInput, ModelPickerScreen

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
_MANUAL_SUBAGENT_FOLLOWUP_PROMPT = (
    "A manual /subagent launch just occurred.\n"
    "Continue orchestration for all currently active workers.\n"
    "Do not launch new workers unless explicitly required for safety.\n"
    "Keep this response open until all workers complete.\n"
    "Provide concise progress updates and then a final combined outcome."
)


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
        and str(worker.get("status", "")) in _ACTIVE_WORKER_STATUSES
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
        self._subagent_runtime: SubAgentRuntime | None = None

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

        self._subagent_runtime = SubAgentRuntime(
            self._settings,
            build_subagent=lambda settings: build_agent_graph(
                settings,
                role="subagent",
            ),
        )
        role = "orchestrator" if self._settings.sub_agent_mode_enabled else "default"
        try:
            agent = build_agent_graph(
                self._settings,
                role=role,
                subagent_runtime=self._subagent_runtime,
            )
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
            subagent_runtime=self._subagent_runtime,
        )
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
        if self._ctx is not None and self._ctx.settings.sub_agent_mode_enabled:
            banner.append("  ", style="dim")
            banner.append("sub-agent mode: ", style="dim")
            banner.append("ON", style="bold green")
        self.query_one("#banner", Static).update(banner)

    def on_unmount(self) -> None:
        if self._subagent_runtime is not None:
            self._subagent_runtime.shutdown()

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
                await self._run_command(command, args)
                return

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
        spawned_subagent_id = str(result.data.get("spawned_subagent_id", "")).strip()
        if (
            spawned_subagent_id
            and self._ctx.settings.sub_agent_mode_enabled
            and self._ctx.subagent_runtime is not None
        ):
            report = self._ctx.subagent_runtime.get_subagent_report()
            try:
                active_count = int(report.get("active_count", 0) or 0)
            except Exception:  # noqa: BLE001
                active_count = 0
            if active_count > 0:
                followup_bubble = await self._append_message(
                    "agent",
                    "",
                    model_name=self._ctx.settings.model_name,
                )
                await self._stream_agent_response(
                    _MANUAL_SUBAGENT_FOLLOWUP_PROMPT,
                    followup_bubble,
                )
        self._refresh_banner()
        if result.should_exit:
            self.exit()

    async def _stream_agent_response(
        self, user_input: str, bubble: ChatBubble
    ) -> None:
        if self._ctx is None:
            return

        md_stream = Markdown.get_stream(bubble.markdown_widget())
        fallback_text = ""
        saw_text = False
        seen_ids: set[str] = set()
        seen_tool_result_ids: set[str] = set()
        reasoning = _ReasoningTracker()
        tool_idx = 0
        agent_status_stop = asyncio.Event()
        agent_status_task: asyncio.Task[None] | None = None
        if (
            self._ctx.settings.sub_agent_mode_enabled
            and self._ctx.subagent_runtime is not None
        ):
            agent_status_task = asyncio.create_task(
                self._poll_agent_status(bubble, agent_status_stop)
            )

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

                user_facing_results = collect_new_user_facing_tool_results(
                    chunk, seen_tool_result_ids
                )
                for result in user_facing_results:
                    bubble.append_tool_output(
                        render_user_facing_tool_output(
                            result["tool_name"],
                            result["render_mode"],
                            result["content"],
                        )
                    )

            await md_stream.stop()

            if reasoning.active:
                bubble.set_reasoning_summary(reasoning.finish())

            if not saw_text and fallback_text.strip():
                await bubble.set_markdown(fallback_text)
        except asyncio.CancelledError:
            await md_stream.stop()
            raise
        except Exception as exc:  # noqa: BLE001
            await md_stream.stop()
            await bubble.set_markdown(f"Request failed: `{exc!s}`")
        finally:
            agent_status_stop.set()
            if agent_status_task is not None:
                with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                    await asyncio.wait_for(agent_status_task, timeout=1.5)
                if not agent_status_task.done():
                    agent_status_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await agent_status_task
            bubble.clear_tool_status()
            bubble.clear_agent_status()
            await self._finish_turn()

    async def _poll_agent_status(
        self,
        bubble: ChatBubble,
        stop_signal: asyncio.Event,
    ) -> None:
        while not stop_signal.is_set():
            status_text = ""
            try:
                if self._ctx is not None and self._ctx.subagent_runtime is not None:
                    report = self._ctx.subagent_runtime.get_subagent_report()
                    status_text = _format_agent_status(report)
            except Exception:  # noqa: BLE001
                status_text = ""
            bubble.update_agent_status(status_text)
            try:
                await asyncio.wait_for(
                    stop_signal.wait(),
                    timeout=_AGENT_STATUS_POLL_INTERVAL_S,
                )
            except asyncio.TimeoutError:
                continue

    async def _finish_turn(self) -> None:
        turn_input = self._turn_input
        turn_output = self._turn_output
        text_len = self._turn_text_len
        self._turn_input = 0
        self._turn_output = 0
        self._turn_text_len = 0

        if turn_input == 0 and turn_output == 0 and text_len > 0:
            turn_output = max(1, text_len // 4)

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
            for hook in self._hooks:
                try:
                    message = await hook.after_turn(hook_ctx)
                except Exception as exc:  # noqa: BLE001
                    message = f"Hook failed: `{exc!s}`"
                if message:
                    await self._append_system(message)
        self._refresh_banner()

    def _extract_usage(self, msg: Any) -> None:
        usage = getattr(msg, "usage_metadata", None)
        if not usage:
            return
        if isinstance(usage, dict):
            self._turn_input += usage.get("input_tokens", 0) or 0
            self._turn_output += usage.get("output_tokens", 0) or 0
        else:
            self._turn_input += getattr(usage, "input_tokens", 0) or 0
            self._turn_output += getattr(usage, "output_tokens", 0) or 0

    @work(exclusive=True, exit_on_error=False)
    async def _load_available_models(self) -> None:
        models = await fetch_models(
            base_url=self._settings.openai_base_url,
            api_key=self._settings.openai_api_key,
        )
        self._available_models = models
        self._refresh_banner()
