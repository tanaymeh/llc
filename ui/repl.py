import asyncio
import uuid
from typing import Any

from langchain_core.messages import HumanMessage

from agent import build_agent_graph
from commands import CommandRegistry, ReplContext
from config import Settings
from ui.colors import BOLD, CYAN, DIM, GREEN, RED, YELLOW, style
from ui.display import message_text, print_banner, print_tool_calls


class Repl:
    def __init__(self, settings: Settings, registry: CommandRegistry) -> None:
        self._registry = registry
        self._settings = settings

    async def run(self) -> None:
        print_banner(self._settings.model_name)

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
                user_input = await asyncio.to_thread(
                    input, f"{style('❯', GREEN, BOLD)} "
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

    async def _run_turn(
        self, agent: Any, thread_id: str, user_input: str
    ) -> None:
        assistant_line_open = False
        seen_tool_call_ids: set[str] = set()

        async for mode, chunk in agent.astream(
            {"messages": [HumanMessage(content=user_input)]},
            config={"configurable": {"thread_id": thread_id}},
            stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                message_chunk, metadata = chunk
                if metadata.get("langgraph_node") != "llm":
                    continue

                text = message_text(message_chunk.content)
                if text:
                    if not assistant_line_open:
                        print(f"\n{style('●', CYAN)} ", end="", flush=True)
                        assistant_line_open = True
                    print(text, end="", flush=True)

            elif mode == "updates":
                assistant_line_open = print_tool_calls(
                    chunk=chunk,
                    assistant_line_open=assistant_line_open,
                    seen_tool_call_ids=seen_tool_call_ids,
                )

        if assistant_line_open:
            print()
        print()
