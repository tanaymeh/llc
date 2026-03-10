from langchain_core.tools import BaseTool

from llc.agent.tools.code_grep import make_code_grep_tools
from llc.agent.tools.diff import make_diff_tools
from llc.agent.tools.edit import make_edit_tools
from llc.agent.tools.filesystem import make_filesystem_tools
from llc.agent.tools.glob import make_glob_tools
from llc.agent.tools.grep import make_grep_tools
from llc.agent.tools.shell import make_shell_tools
from llc.agent.tools.todo import make_todo_tools
from llc.agent.tools.web import make_web_tools
from llc.config import Settings


def collect_tools(settings: Settings) -> list[BaseTool]:
    return [
        *make_filesystem_tools(settings.workspace_root),
        *make_shell_tools(settings.shell_timeout),
        *make_glob_tools(settings.workspace_root),
        *make_grep_tools(settings.workspace_root),
        *make_edit_tools(settings.workspace_root),
        *make_diff_tools(settings.workspace_root),
        *make_todo_tools(),
        *make_web_tools(settings.firecrawl_api_key),
        *make_code_grep_tools(settings.workspace_root),
    ]
