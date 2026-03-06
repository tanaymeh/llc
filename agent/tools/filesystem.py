from pathlib import Path

from langchain.tools import tool
from langchain_core.tools import BaseTool


def _resolve_path(workspace_root: Path, file_path: str) -> Path:
    candidate = (workspace_root / file_path).resolve()
    try:
        candidate.relative_to(workspace_root)
    except ValueError:
        raise ValueError("Path is outside the workspace root and is not allowed.")
    return candidate


def make_filesystem_tools(workspace_root: Path) -> list[BaseTool]:
    @tool
    def read_file(file_path: str) -> str:
        """Read a text file from the current workspace."""
        target = _resolve_path(workspace_root, file_path)
        if not target.exists():
            return f"File does not exist: {file_path}"
        if target.is_dir():
            return f"Path is a directory, not a file: {file_path}"
        return target.read_text(encoding="utf-8")

    @tool
    def write_file(file_path: str, content: str) -> str:
        """Write UTF-8 text content to a file in the workspace."""
        target = _resolve_path(workspace_root, file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to {file_path}"

    @tool
    def list_directory(path: str = ".") -> str:
        """List files and folders in the given workspace path."""
        target = _resolve_path(workspace_root, path)
        if not target.exists():
            return f"Path does not exist: {path}"
        if not target.is_dir():
            return f"Path is not a directory: {path}"

        entries = sorted(
            target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
        )
        if not entries:
            return f"No entries in {path}"

        lines: list[str] = []
        for entry in entries:
            marker = "/" if entry.is_dir() else ""
            rel = entry.relative_to(workspace_root)
            lines.append(f"{rel}{marker}")
        return "\n".join(lines)

    return [read_file, write_file, list_directory]
