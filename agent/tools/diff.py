import subprocess
from pathlib import Path

from langchain.tools import tool
from langchain_core.tools import BaseTool


def make_diff_tools(workspace_root: Path) -> list[BaseTool]:
    @tool
    def ShowDiff() -> str:
        """Shows a color-coded diff of all changed files in the workspace (uncommitted changes).

Use this after every successful implementation so the user can see what was changed, or when the user asks to see the diff. The output uses ANSI colors: green for additions, red for removals. If the workspace is not a git repo or there are no changes, returns a short message instead."""
        try:
            staged = subprocess.run(
                ["git", "diff", "--cached", "--color=always"],
                cwd=workspace_root,
                capture_output=True,
                text=True,
                timeout=30,
            )
            unstaged = subprocess.run(
                ["git", "diff", "--color=always"],
                cwd=workspace_root,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return "Diff timed out."
        except FileNotFoundError:
            return "Not a git repository or git not available."

        staged_out = (staged.stdout or "").strip()
        unstaged_out = (unstaged.stdout or "").strip()

        if not staged_out and not unstaged_out:
            return "No uncommitted changes."

        parts = []
        if staged_out:
            parts.append("=== Staged changes ===\n" + staged_out)
        if unstaged_out:
            parts.append("=== Unstaged changes ===\n" + unstaged_out)
        return "\n\n".join(parts)

    return [ShowDiff]
