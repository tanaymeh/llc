from agent.tools.filesystem import list_directory, read_file, write_file
from agent.tools.shell import run_shell_command

TOOLS = [run_shell_command, read_file, write_file, list_directory]
