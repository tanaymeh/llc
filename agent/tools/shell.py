import subprocess

from langchain.tools import tool
from langchain_core.tools import BaseTool


def make_shell_tools(timeout: int = 30) -> list[BaseTool]:
    @tool
    def run_shell_command(command: str) -> str:
        """Run a shell command in the current workspace and return output."""
        try:
            completed = subprocess.run(
                command,
                shell=True,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout} seconds."

        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()

        result_parts = [f"Exit code: {completed.returncode}"]
        if stdout:
            result_parts.append(f"STDOUT:\n{stdout}")
        if stderr:
            result_parts.append(f"STDERR:\n{stderr}")

        return "\n\n".join(result_parts)

    return [run_shell_command]
