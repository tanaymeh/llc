import argparse

from llc.commands import CommandRegistry
from llc.commands.compact import CompactCommand
from llc.commands.enable import EnableCommand
from llc.commands.exit import ExitCommand
from llc.commands.help import HelpCommand
from llc.commands.model import ModelCommand
from llc.commands.subagent import SubagentCommand
from llc.config import Settings
from llc.logging_utils import configure_logging
from llc.service.api import create_api_app


def _build_registry() -> CommandRegistry:
    registry = CommandRegistry()
    registry.register(CompactCommand())
    registry.register(ExitCommand())
    registry.register(ModelCommand())
    registry.register(EnableCommand())
    registry.register(SubagentCommand())
    registry.register(HelpCommand(registry))
    return registry


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
        description="Run the LLC API server.",
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("serve",),
        default="serve",
        help="Optional explicit API mode for backward compatibility.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Override API host.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override API port.",
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
    configure_logging(subagent_debug_logging=settings.sub_agent_debug_logging)
    registry = _build_registry()
    _run_api(settings, registry)


if __name__ == "__main__":
    main()
