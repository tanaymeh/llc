from pathlib import Path
from typing import Optional

from langchain.tools import tool
from langchain_core.tools import BaseTool


def make_glob_tools(workspace_root: Path) -> list[BaseTool]:
    @tool
    def Glob(pattern: str, path: Optional[str] = None) -> str:
        """- Fast file pattern matching tool that works with any codebase size
- Supports glob patterns like "**/*.js" or "src/**/*.ts"
- Returns matching file paths sorted by modification time
- Use this tool when you need to find files by name patterns
- When you are doing an open ended search that may require multiple rounds of globbing and grepping, use the Agent tool instead
- You have the capability to call multiple tools in a single response. It is always better to speculatively perform multiple searches as a batch that are potentially useful."""
        search_root = workspace_root
        if path:
            candidate = (workspace_root / path).resolve()
            try:
                candidate.relative_to(workspace_root)
            except ValueError:
                return "Error: path is outside the workspace root."
            if not candidate.is_dir():
                return f"Error: not a directory: {path}"
            search_root = candidate

        try:
            matches = list(search_root.glob(pattern))
        except Exception as exc:
            return f"Error in glob pattern: {exc}"

        if not matches:
            return "No matches found."

        file_matches = [m for m in matches if m.is_file()]
        if not file_matches:
            return "No file matches found."

        file_matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        lines: list[str] = []
        for m in file_matches:
            try:
                rel = m.relative_to(workspace_root)
            except ValueError:
                rel = m
            lines.append(str(rel))

        return "\n".join(lines)

    return [Glob]
