from agent import build_agent_graph
from commands import Command, CommandResult, ReplContext
from ui.display import ACCENT_STYLE, MUTED_STYLE, console


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
                message="  [dim]Usage:[/dim] [bold]/model <model_name>[/bold]"
            )

        console.print(
            f"  [{MUTED_STYLE}]Switching to[/{MUTED_STYLE}] "
            f"[bold {ACCENT_STYLE}]{args}[/bold {ACCENT_STYLE}]"
            f"[{MUTED_STYLE}]...[/{MUTED_STYLE}]"
        )
        try:
            new_settings = ctx.settings.model_copy(update={"model_name": args})
            ctx.agent = build_agent_graph(new_settings)
            ctx.settings = new_settings
            return CommandResult(
                message=f"  [green]✓ Now using {args}[/green]"
            )
        except Exception as exc:  # noqa: BLE001
            return CommandResult(
                message=f"  [red]✗ {exc!s}[/red]"
            )
