from llc.agent.compact import aget_history_messages, compact_history
from llc.commands import Command, CommandResult, ReplContext


class CompactCommand(Command):
    @property
    def name(self) -> str:
        return "/compact"

    @property
    def description(self) -> str:
        return "Summarize and replace older chat history"

    async def execute(self, args: str, ctx: ReplContext) -> CommandResult:
        if args.strip():
            return CommandResult(message="Usage: `/compact`")

        messages = await aget_history_messages(ctx.agent, ctx.thread_id)
        if len(messages) <= 5:
            return CommandResult(
                message="Need more than 5 chat messages before compaction."
            )

        compacted_count = await compact_history(
            ctx.agent,
            ctx.thread_id,
            ctx.settings,
        )
        if compacted_count <= 0:
            return CommandResult(message="Nothing to compact right now.")

        return CommandResult(
            message=f"Compacted {compacted_count} messages into a summary."
        )
