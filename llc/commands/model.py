from llc.agent import build_agent_graph
from llc.commands import Command, CommandResult, ReplContext


class ModelCommand(Command):
    @property
    def name(self) -> str:
        return "/model"

    @property
    def description(self) -> str:
        return "Switch to a different model"

    async def execute(self, args: str, ctx: ReplContext) -> CommandResult:
        model_name = args.strip()
        if not model_name:
            return CommandResult(
                message="Usage: `/model <model_name>`"
            )

        try:
            new_settings = ctx.settings.model_copy(update={"model_name": model_name})
            ctx.agent = build_agent_graph(new_settings)
            ctx.settings = new_settings
            return CommandResult(
                message=f"Now using `{model_name}`."
            )
        except Exception as exc:  # noqa: BLE001
            return CommandResult(
                message=f"Failed to switch model: {exc!s}"
            )
