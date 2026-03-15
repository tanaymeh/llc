from llc.agent import build_agent_graph
from llc.commands import Command, CommandResult, ReplContext


class EnableCommand(Command):
    @property
    def name(self) -> str:
        return "/enable"

    @property
    def description(self) -> str:
        return "Enable an experimental runtime mode"

    async def execute(self, args: str, ctx: ReplContext) -> CommandResult:
        mode = args.strip()
        if mode != "sub-agent-mode":
            return CommandResult(
                message="Usage: `/enable sub-agent-mode`"
            )

        if ctx.settings.sub_agent_mode_enabled:
            return CommandResult(message="`sub-agent-mode` is already enabled.")

        try:
            new_settings = ctx.settings.model_copy(update={"sub_agent_mode_enabled": True})
            if ctx.subagent_runtime is not None:
                ctx.subagent_runtime.update_settings(new_settings)
            ctx.agent = build_agent_graph(
                new_settings,
                role="orchestrator",
                subagent_runtime=ctx.subagent_runtime,
                prompt_registry=ctx.prompt_registry,
            )
            ctx.settings = new_settings
            return CommandResult(
                message=(
                    "Enabled `sub-agent-mode`. "
                    "Use `/subagent {TASK}` to spawn workers manually."
                )
            )
        except Exception as exc:  # noqa: BLE001
            return CommandResult(message=f"Failed to enable sub-agent mode: {exc!s}")
