import subprocess
from pathlib import Path
from typing import Optional

from langchain.tools import tool
from langchain_core.tools import BaseTool


def make_grep_tools(workspace_root: Path) -> list[BaseTool]:
    @tool
    def Grep(
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
        output_mode: Optional[str] = None,
        before_context: Optional[int] = None,
        after_context: Optional[int] = None,
        context: Optional[int] = None,
        line_numbers: Optional[bool] = None,
        case_insensitive: Optional[bool] = None,
        type: Optional[str] = None,
        head_limit: Optional[int] = None,
        multiline: Optional[bool] = None,
    ) -> str:
        """A powerful search tool built on ripgrep

  Usage:
  - ALWAYS use Grep for search tasks. NEVER invoke `grep` or `rg` as a Bash command. The Grep tool has been optimized for correct permissions and access.
  - Supports full regex syntax (e.g., "log.*Error", "function\\s+\\w+")
  - Filter files with glob parameter (e.g., "*.js", "**/*.tsx") or type parameter (e.g., "js", "py", "rust")
  - Output modes: "content" shows matching lines, "files_with_matches" shows only file paths (default), "count" shows match counts
  - Use Task tool for open-ended searches requiring multiple rounds
  - Pattern syntax: Uses ripgrep (not grep) - literal braces need escaping (use `interface\\{\\}` to find `interface{}` in Go code)
  - Multiline matching: By default patterns match within single lines only. For cross-line patterns like `struct \\{[\\s\\S]*?field`, use `multiline: true`"""
        cmd: list[str] = ["rg"]

        mode = output_mode or "files_with_matches"
        if mode == "files_with_matches":
            cmd.append("--files-with-matches")
        elif mode == "count":
            cmd.append("--count")

        if case_insensitive:
            cmd.append("-i")

        if multiline:
            cmd.extend(["-U", "--multiline-dotall"])

        if mode == "content":
            if line_numbers:
                cmd.append("-n")
            if before_context is not None:
                cmd.extend(["-B", str(before_context)])
            if after_context is not None:
                cmd.extend(["-A", str(after_context)])
            if context is not None:
                cmd.extend(["-C", str(context)])

        if glob:
            cmd.extend(["--glob", glob])

        if type:
            cmd.extend(["--type", type])

        cmd.extend(["--", pattern])

        search_path = str(workspace_root)
        if path:
            candidate = (workspace_root / path).resolve()
            try:
                candidate.relative_to(workspace_root)
            except ValueError:
                return "Error: path is outside the workspace root."
            search_path = str(candidate)

        cmd.append(search_path)

        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(workspace_root),
            )
        except FileNotFoundError:
            return "Error: ripgrep (rg) is not installed. Install it with: brew install ripgrep"
        except subprocess.TimeoutExpired:
            return "Search timed out after 30 seconds."

        output = completed.stdout.strip()
        if not output:
            if completed.returncode == 1:
                return "No matches found."
            if completed.returncode != 0 and completed.stderr:
                return f"Error: {completed.stderr.strip()}"
            return "No matches found."

        if head_limit is not None and head_limit > 0:
            lines = output.split("\n")
            output = "\n".join(lines[:head_limit])

        return output

    return [Grep]
