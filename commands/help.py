from commands import Command, CommandRegistry, CommandResult, ReplContext


class HelpCommand(Command):
    def __init__(self, registry: CommandRegistry) -> None:
        self._registry = registry

    @property
    def name(self) -> str:
        return "/help"

    @property
    def description(self) -> str:
        return "Show available commands"

    async def execute(self, args: str, ctx: ReplContext) -> CommandResult:
        lines = ["  Available commands:"]
        for cmd in self._registry.all_commands:
            names = [cmd.name, *cmd.aliases]
            label = ", ".join(f"[bold]{n}[/bold]" for n in names)
            lines.append(f"    {label}  [dim]{cmd.description}[/dim]")
        return CommandResult(message="\n".join(lines))
