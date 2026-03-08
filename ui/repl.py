import asyncio
import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.shortcuts import CompleteStyle, PromptSession

from agent import build_agent_graph
from commands import CommandRegistry, ReplContext
from config import Settings
from models import AvailableModel, fetch_models
from ui.colors import BOLD, CYAN, DIM, GREEN, RED, YELLOW, style
from ui.completer import ModelCompleter
from ui.display import (
    ReasoningTracker,
    extract_reasoning,
    message_text,
    print_banner,
    print_tool_calls,
)


class Repl:
    def __init__(self, settings: Settings, registry: CommandRegistry) -> None:
        self._registry = registry
        self._settings = settings
        self._available_models: list[AvailableModel] = []
        self._prompt_session = PromptSession(
            completer=ModelCompleter(self._get_available_models),
            complete_while_typing=True,
            complete_style=CompleteStyle.COLUMN,
            reserve_space_for_menu=8,
        )

    async def run(self) -> None:
        print_banner(self._settings.model_name)
        asyncio.create_task(self._load_available_models())

        try:
            agent = build_agent_graph(self._settings)
        except Exception as exc:  # noqa: BLE001
            print(
                f"\n{style('✗', RED)} "
                f"{style('Failed to initialize model.', RED, BOLD)}"
            )
            print(
                f"  {style('Set MODEL_NAME, OPENAI_API_KEY, and optionally OPENAI_BASE_URL', DIM)}"
                f"\n  {style('in your environment or .env file.', DIM)}"
            )
            print(f"  {style(str(exc), RED)}")
            return

        ctx = ReplContext(settings=self._settings, agent=agent)
        thread_id = f"cli-{uuid.uuid4()}"

        while True:
            try:
                user_input = await self._prompt_session.prompt_async(
                    ANSI(f"{style('❯', GREEN, BOLD)} ")
                )
                user_input = user_input.strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n{style('Goodbye.', DIM)}")
                break

            if not user_input:
                continue

            match = self._registry.match(user_input)
            if match:
                cmd, args = match
                result = await cmd.execute(args, ctx)
                if result.message:
                    print(result.message)
                    print()
                if result.should_exit:
                    print(style("Goodbye.", DIM))
                    break
                continue

            try:
                await self._run_turn(ctx.agent, thread_id, user_input)
            except KeyboardInterrupt:
                print(f"\n{style('Interrupted.', YELLOW)}\n")
            except Exception as exc:  # noqa: BLE001
                print(f"\n{style('✗', RED)} {style(str(exc), RED)}\n")

    def _get_available_models(self) -> list[AvailableModel]:
        return self._available_models

    async def _load_available_models(self) -> None:
        self._available_models = await fetch_models(
            base_url=self._settings.openai_base_url,
            api_key=self._settings.openai_api_key,
        )

    async def _run_turn(
        self, agent: Any, thread_id: str, user_input: str
    ) -> None:
        assistant_line_open = False
        displayed_assistant_text = False
        seen_tool_call_ids: set[str] = set()
        fallback_assistant_text = ""
        reasoning_tracker = ReasoningTracker()

        async for mode, chunk in agent.astream(
            {"messages": [HumanMessage(content=user_input)]},
            config={"configurable": {"thread_id": thread_id}},
            stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                message_chunk, metadata = chunk
                if metadata.get("langgraph_node") != "llm":
                    continue

                reasoning_text = extract_reasoning(message_chunk)
                if reasoning_text and not assistant_line_open:
                    reasoning_tracker.feed(reasoning_text)
                    continue

                text = message_text(message_chunk.content)
                if text:
                    finished_reasoning = False
                    if reasoning_tracker.is_active:
                        reasoning_tracker.finish()
                        finished_reasoning = True

                    if not assistant_line_open:
                        prefix = "" if finished_reasoning else "\n"
                        print(f"{prefix}{style('●', CYAN)} ", end="", flush=True)
                        assistant_line_open = True
                    print(text, end="", flush=True)
                    displayed_assistant_text = True

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
                            if getattr(message, "tool_calls", None):
                                continue
                            text = message_text(message.content)
                            if text:
                                fallback_assistant_text = text
                assistant_line_open = print_tool_calls(
                    chunk=chunk,
                    assistant_line_open=assistant_line_open,
                    seen_tool_call_ids=seen_tool_call_ids,
                )

        if reasoning_tracker.is_active:
            reasoning_tracker.finish()
        if not displayed_assistant_text and fallback_assistant_text:
            if assistant_line_open:
                print()
                assistant_line_open = False
            print(f"\n{style('●', CYAN)} {fallback_assistant_text}")
        if assistant_line_open:
            print()
        print()
