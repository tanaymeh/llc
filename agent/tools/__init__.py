from langchain_core.tools import BaseTool

from agent.tools.filesystem import make_filesystem_tools
from agent.tools.shell import make_shell_tools
from config import Settings


def collect_tools(settings: Settings) -> list[BaseTool]:
    return [
        *make_filesystem_tools(settings.workspace_root),
        *make_shell_tools(settings.shell_timeout),
    ]
