from llc.commands import CommandRegistry
from llc.commands.compact import CompactCommand
from llc.commands.enable import EnableCommand
from llc.commands.exit import ExitCommand
from llc.commands.help import HelpCommand
from llc.commands.model import ModelCommand
from llc.commands.subagent import SubagentCommand
from llc.config import Settings
from llc.service.engine import SessionEngine
from llc.storage.store import SessionStore
from llc.ui import MissionControlApp


def _build_registry() -> CommandRegistry:
    registry = CommandRegistry()
    registry.register(CompactCommand())
    registry.register(ExitCommand())
    registry.register(ModelCommand())
    registry.register(EnableCommand())
    registry.register(SubagentCommand())
    registry.register(HelpCommand(registry))
    return registry


def main() -> None:
    settings = Settings.from_env()
    registry = _build_registry()
    store = SessionStore(settings.db_path)
    engine = SessionEngine(settings, registry, store=store)
    app = MissionControlApp(engine)
    app.run()


if __name__ == "__main__":
    main()
