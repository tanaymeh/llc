import asyncio

from commands import CommandRegistry
from commands.exit import ExitCommand
from commands.help import HelpCommand
from commands.model import ModelCommand
from config import Settings
from ui.repl import Repl


def _build_registry() -> CommandRegistry:
    registry = CommandRegistry()
    registry.register(ExitCommand())
    registry.register(ModelCommand())
    registry.register(HelpCommand(registry))
    return registry


def main() -> None:
    settings = Settings.from_env()
    registry = _build_registry()
    repl = Repl(settings, registry)
    asyncio.run(repl.run())


if __name__ == "__main__":
    main()
