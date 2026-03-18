import argparse

from llc.commands import CommandRegistry
from llc.commands.compact import CompactCommand
from llc.commands.enable import EnableCommand
from llc.commands.exit import ExitCommand
from llc.commands.help import HelpCommand
from llc.commands.model import ModelCommand
from llc.commands.subagent import SubagentCommand
from llc.config import Settings
from llc.service.engine import SessionEngine
from llc.service.api import create_api_app
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


def _run_tui(settings: Settings, registry: CommandRegistry) -> None:
    store = SessionStore(settings.db_path)
    engine = SessionEngine(settings, registry, store=store)
    app = MissionControlApp(engine)
    app.run()


def _run_api(settings: Settings, registry: CommandRegistry) -> None:
    import uvicorn

    app = create_api_app(settings, registry)
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level="info",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="llc",
        description="Run LLC API server (default) or Textual TUI.",
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("serve", "tui"),
        default="serve",
        help="`serve` starts the API server, `tui` starts the terminal UI.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Override API host (serve mode only).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override API port (serve mode only).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = Settings.from_env()
    if args.host or args.port:
        updates = {
            key: value
            for key, value in {
                "api_host": args.host,
                "api_port": args.port,
            }.items()
            if value is not None
        }
        if updates:
            settings = settings.model_copy(update=updates)
    registry = _build_registry()
    if args.mode == "tui":
        _run_tui(settings, registry)
        return
    _run_api(settings, registry)


if __name__ == "__main__":
    main()
