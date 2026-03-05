import subprocess

from langchain.tools import tool


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
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return "Command timed out after 30 seconds."

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()

    result_parts = [f"Exit code: {completed.returncode}"]
    if stdout:
        result_parts.append(f"STDOUT:\n{stdout}")
    if stderr:
        result_parts.append(f"STDERR:\n{stderr}")

    return "\n\n".join(result_parts)
