from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llc.agent.subagents import SubAgentRuntime
    from llc.config import Settings


class CommandResult:
    def __init__(
        self,
        should_exit: bool = False,
        message: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.should_exit = should_exit
        self.message = message
        self.data = data or {}


class ReplContext:
    def __init__(
        self,
        settings: Settings,
        agent: Any,
        thread_id: str,
        subagent_runtime: "SubAgentRuntime | None" = None,
    ) -> None:
        self.settings = settings
        self.agent = agent
        self.thread_id = thread_id
        self.subagent_runtime = subagent_runtime


class Command(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def aliases(self) -> list[str]:
        return []

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    async def execute(self, args: str, ctx: ReplContext) -> CommandResult: ...


class CommandRegistry:
    def __init__(self) -> None:
        self._commands: dict[str, Command] = {}
        self._unique: list[Command] = []

    def register(self, command: Command) -> None:
        self._commands[command.name] = command
        self._unique.append(command)
        for alias in command.aliases:
            self._commands[alias] = command

    def match(self, user_input: str) -> tuple[Command, str] | None:
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = self._commands.get(parts[0])
            if cmd:
                return cmd, (parts[1].strip() if len(parts) > 1 else "")

        word = user_input.strip().lower()
        cmd = self._commands.get(word)
        if cmd:
            return cmd, ""

        return None

    @property
    def all_commands(self) -> list[Command]:
        return list(self._unique)
