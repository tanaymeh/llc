import asyncio
import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.shortcuts import CompleteStyle, PromptSession
from rich.live import Live
from rich.markdown import Markdown
from rich.spinner import Spinner
from rich.text import Text

from agent import build_agent_graph
from commands import CommandRegistry, ReplContext
from config import Settings
from models import AvailableModel, fetch_models, get_model_pricing
from ui.colors import BOLD, GREEN, style
from ui.completer import ModelCompleter
from ui.display import (
    MARKDOWN_CODE_THEME,
    MUTED_STYLE,
    ReasoningTracker,
    TOOL_NAME_STYLE,
    collect_new_tool_calls,
    console,
    extract_reasoning,
    format_tool_args,
    message_text,
    print_banner,
    print_session_summary,
    render_session_hud,
    render_assistant_markdown,
)
from ui.token_tracker import TokenTracker


class Repl:
    def __init__(self, settings: Settings, registry: CommandRegistry) -> None:
        self._registry = registry
        self._settings = settings
        self._available_models: list[AvailableModel] = []
        self._token_tracker = TokenTracker()
        self._session_cost: float = 0.0
        self._prompt_session = PromptSession(
            completer=ModelCompleter(self._get_available_models),
            complete_while_typing=True,
            complete_style=CompleteStyle.COLUMN,
            reserve_space_for_menu=8,
        )

    async def run(self) -> None:
        print_banner(self._settings.model_name)
        render_session_hud(0, 0, None)
        asyncio.create_task(self._load_available_models())

        try:
            agent = build_agent_graph(self._settings)
        except Exception as exc:  # noqa: BLE001
            console.print(
                f"\n[red bold]✗ Failed to initialize model.[/red bold]"
            )
            console.print(
                f"  [dim]Set MODEL_NAME, OPENAI_API_KEY, and optionally OPENAI_BASE_URL[/dim]"
                f"\n  [dim]in your environment or .env file.[/dim]"
            )
            console.print(f"  [red]{exc!s}[/red]")
            return

        ctx = ReplContext(settings=self._settings, agent=agent)
        thread_id = f"cli-{uuid.uuid4()}"

        while True:
            try:
                render_session_hud(
                    self._token_tracker.session_input_tokens,
                    self._token_tracker.session_output_tokens,
                    self._session_cost if self._session_cost > 0 else None,
                )
                user_input = await self._prompt_session.prompt_async(
                    ANSI(f"{style('❯', GREEN, BOLD)} ")
                )
                user_input = user_input.strip()
            except (EOFError, KeyboardInterrupt):
                self._print_goodbye()
                break

            if not user_input:
                continue

            match = self._registry.match(user_input)
            if match:
                cmd, args = match
                result = await cmd.execute(args, ctx)
                if result.message:
                    console.print(result.message)
                    console.print()
                if result.should_exit:
                    self._print_goodbye()
                    break
                continue

            try:
                await self._run_turn(ctx.agent, thread_id, user_input)
            except KeyboardInterrupt:
                console.print(f"\n[yellow]Interrupted.[/yellow]\n")
            except Exception as exc:  # noqa: BLE001
                console.print(f"\n[red]✗ {exc!s}[/red]\n")

    def _print_goodbye(self) -> None:
        console.print()
        if (
            self._token_tracker.session_input_tokens > 0
            or self._token_tracker.session_output_tokens > 0
        ):
            cost = self._session_cost if self._session_cost > 0 else None
            print_session_summary(
                self._token_tracker.session_input_tokens,
                self._token_tracker.session_output_tokens,
                cost,
            )
        console.print("[dim]Goodbye.[/dim]")

    def _get_available_models(self) -> list[AvailableModel]:
        return self._available_models

    async def _load_available_models(self) -> None:
        self._available_models = await fetch_models(
            base_url=self._settings.openai_base_url,
            api_key=self._settings.openai_api_key,
        )

    def _stop_live(self, live: Live | None) -> None:
        if live is not None:
            live.stop()

    def _stop_spinner(self, spinner: Live | None) -> None:
        if spinner is not None:
            spinner.stop()

    def _flush_buffer(self, buffer: str, live: Live | None) -> tuple[str, Live | None]:
        """Stop live preview and render buffer as a bordered panel."""
        self._stop_live(live)
        if buffer.strip():
            render_assistant_markdown(buffer)
        return "", None

    def _tool_spinner_renderable(
        self,
        name: str,
        args: dict[str, Any],
        position: int,
        total: int,
    ) -> Spinner:
        args_str = format_tool_args(args)
        spinner_text = Text()
        spinner_text.append("  ↳ ", style=MUTED_STYLE)
        spinner_text.append(f"[{position}/{total}] ", style=MUTED_STYLE)
        spinner_text.append(name, style=TOOL_NAME_STYLE)
        spinner_text.append("(", style=MUTED_STYLE)
        spinner_text.append(args_str, style=MUTED_STYLE)
        spinner_text.append(")", style=MUTED_STYLE)
        return Spinner("dots", text=spinner_text)

    def _start_spinner(
        self,
        name: str,
        args: dict[str, Any],
        position: int,
        total: int,
    ) -> Live:
        sp = Live(
            self._tool_spinner_renderable(name, args, position, total),
            console=console,
            refresh_per_second=12,
            transient=True,
        )
        sp.start()
        return sp

    def _update_spinner(
        self,
        spinner: Live,
        name: str,
        args: dict[str, Any],
        position: int,
        total: int,
    ) -> None:
        spinner.update(self._tool_spinner_renderable(name, args, position, total))

    @staticmethod
    def _cancel_task(task: asyncio.Task[None] | None) -> None:
        if task is not None and not task.done():
            task.cancel()

    async def _run_turn(
        self, agent: Any, thread_id: str, user_input: str
    ) -> None:
        assistant_buffer = ""
        fallback_text = ""
        seen_tool_call_ids: set[str] = set()
        reasoning_tracker = ReasoningTracker()
        live: Live | None = None
        tool_spinner: Live | None = None
        tool_queue: list[tuple[str, dict[str, Any]]] = []
        tool_roll_index = 0
        tool_roll_task: asyncio.Task[None] | None = None
        any_text_streamed = False

        async def roll_tool_calls() -> None:
            nonlocal tool_roll_index, tool_spinner
            try:
                while True:
                    await asyncio.sleep(0.45)
                    if tool_spinner is None or len(tool_queue) <= 1:
                        continue
                    tool_roll_index = (tool_roll_index + 1) % len(tool_queue)
                    name, args = tool_queue[tool_roll_index]
                    self._update_spinner(
                        tool_spinner,
                        name,
                        args,
                        tool_roll_index + 1,
                        len(tool_queue),
                    )
            except asyncio.CancelledError:
                return

        async for mode, chunk in agent.astream(
            {"messages": [HumanMessage(content=user_input)]},
            config={"configurable": {"thread_id": thread_id}},
            stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                message_chunk, metadata = chunk
                if metadata.get("langgraph_node") != "llm":
                    continue

                self._extract_usage(message_chunk)

                reasoning_text = extract_reasoning(message_chunk)
                if reasoning_text and live is None:
                    if tool_spinner is not None:
                        self._stop_spinner(tool_spinner)
                        tool_spinner = None
                        self._cancel_task(tool_roll_task)
                        tool_roll_task = None
                    reasoning_tracker.feed(reasoning_text)
                    continue

                text = message_text(message_chunk.content)
                if text:
                    if tool_spinner is not None:
                        self._stop_spinner(tool_spinner)
                        tool_spinner = None
                        self._cancel_task(tool_roll_task)
                        tool_roll_task = None

                    if reasoning_tracker.is_active:
                        reasoning_tracker.finish()

                    assistant_buffer += text
                    any_text_streamed = True

                    if live is None:
                        live = Live(
                            Markdown(assistant_buffer, code_theme=MARKDOWN_CODE_THEME),
                            console=console,
                            refresh_per_second=12,
                            vertical_overflow="visible",
                            transient=True,
                        )
                        live.start()
                    else:
                        live.update(
                            Markdown(assistant_buffer, code_theme=MARKDOWN_CODE_THEME)
                        )

            elif mode == "updates":
                if reasoning_tracker.is_active:
                    reasoning_tracker.finish()

                if isinstance(chunk, dict):
                    for node_update in chunk.values():
                        if not isinstance(node_update, dict):
                            continue
                        for message in node_update.get("messages", []):
                            if not isinstance(message, AIMessage):
                                continue
                            self._extract_usage(message)
                            if not getattr(message, "tool_calls", None):
                                text = message_text(message.content)
                                if text:
                                    fallback_text = text

                new_tool_calls = collect_new_tool_calls(chunk, seen_tool_call_ids)
                if new_tool_calls:
                    assistant_buffer, live = self._flush_buffer(
                        assistant_buffer, live
                    )

                    for tc in new_tool_calls:
                        name = tc.get("name", "unknown_tool")
                        args = tc.get("args", {})
                        tool_queue.append((name, args))

                    if tool_queue:
                        tool_roll_index = len(tool_queue) - 1
                        current_name, current_args = tool_queue[tool_roll_index]
                        if tool_spinner is None:
                            tool_spinner = self._start_spinner(
                                current_name,
                                current_args,
                                tool_roll_index + 1,
                                len(tool_queue),
                            )
                            if tool_roll_task is None:
                                tool_roll_task = asyncio.create_task(roll_tool_calls())
                        else:
                            self._update_spinner(
                                tool_spinner,
                                current_name,
                                current_args,
                                tool_roll_index + 1,
                                len(tool_queue),
                            )

        self._cancel_task(tool_roll_task)
        self._stop_spinner(tool_spinner)

        if reasoning_tracker.is_active:
            reasoning_tracker.finish()

        if live is not None or assistant_buffer.strip():
            self._flush_buffer(assistant_buffer, live)
        elif not any_text_streamed and fallback_text.strip():
            render_assistant_markdown(fallback_text)

        turn = self._token_tracker.finish_turn()
        if turn.input_tokens > 0 or turn.output_tokens > 0:
            pricing = get_model_pricing(
                self._available_models, self._settings.model_name
            )
            if pricing:
                self._session_cost += TokenTracker.compute_cost(
                    turn, pricing[0], pricing[1]
                )

            render_session_hud(
                self._token_tracker.session_input_tokens,
                self._token_tracker.session_output_tokens,
                self._session_cost if self._session_cost > 0 else None,
            )

        console.print()

    def _extract_usage(self, message_chunk: Any) -> None:
        usage = getattr(message_chunk, "usage_metadata", None)
        if usage and isinstance(usage, dict):
            in_tok = usage.get("input_tokens", 0)
            out_tok = usage.get("output_tokens", 0)
            if in_tok or out_tok:
                self._token_tracker.add(in_tok, out_tok)
