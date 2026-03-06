from agent import build_agent_graph
from commands import Command, CommandResult, ReplContext
from ui.colors import BOLD, CYAN, DIM, GREEN, RED, style


class ModelCommand(Command):
    @property
    def name(self) -> str:
        return "/model"

    @property
    def description(self) -> str:
        return "Switch to a different model"

    async def execute(self, args: str, ctx: ReplContext) -> CommandResult:
        if not args:
            return CommandResult(
                message=f"  {style('Usage:', DIM)} {style('/model <model_name>', BOLD)}"
            )

        print(
            f"  {style('Switching to', DIM)} "
            f"{style(args, CYAN, BOLD)}{style('...', DIM)}"
        )
        try:
            new_settings = ctx.settings.model_copy(update={"model_name": args})
            ctx.agent = build_agent_graph(new_settings)
            ctx.settings = new_settings
            return CommandResult(
                message=f"  {style('✓', GREEN)} {style(f'Now using {args}', GREEN)}"
            )
        except Exception as exc:  # noqa: BLE001
            return CommandResult(
                message=f"  {style('✗', RED)} {style(str(exc), RED)}"
            )
