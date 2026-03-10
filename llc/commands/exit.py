from llc.commands import Command, CommandResult, ReplContext


class ExitCommand(Command):
    @property
    def name(self) -> str:
        return "exit"

    @property
    def aliases(self) -> list[str]:
        return ["quit"]

    @property
    def description(self) -> str:
        return "Exit the session"

    async def execute(self, args: str, ctx: ReplContext) -> CommandResult:
        return CommandResult(should_exit=True)
