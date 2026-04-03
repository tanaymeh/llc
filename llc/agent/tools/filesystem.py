import fnmatch
from pathlib import Path
from typing import Callable, Optional

from langchain.tools import tool
from langchain_core.tools import BaseTool

from llc.agent.tools._paths import resolve_workspace_path


def make_filesystem_tools(
    workspace_root: Path,
    *,
    mutation_guard: Callable[[str], str | None] | None = None,
) -> list[BaseTool]:
    @tool
    def Read(file_path: str, offset: Optional[int] = None, limit: Optional[int] = None) -> str:
        """Reads a file from the local filesystem. You can access any file directly by using this tool.
Assume this tool is able to read all files on the machine. If the User provides a path to a file assume that path is valid. It is okay to read a file that does not exist; an error will be returned.

Usage:
- The file_path parameter must be an absolute path, not a relative path
- By default, it reads up to 2000 lines starting from the beginning of the file
- You can optionally specify a line offset and limit (especially handy for long files), but it's recommended to read the whole file by not providing these parameters
- Any lines longer than 2000 characters will be truncated
- Results are returned using cat -n format, with line numbers starting at 1
- This tool allows the LLC agent to read images (eg PNG, JPG, etc). When reading an image file the contents are presented visually because the model supports multimodal input.
- This tool can read PDF files (.pdf). PDFs are processed page by page, extracting both text and visual content for analysis.
- This tool can read Jupyter notebooks (.ipynb files) and returns all cells with their outputs, combining code, text, and visualizations.
- You have the capability to call multiple tools in a single response. It is always better to speculatively read multiple files as a batch that are potentially useful.
- You will regularly be asked to read screenshots. If the user provides a path to a screenshot ALWAYS use this tool to view the file at the path. This tool will work with all temporary file paths like /var/folders/123/abc/T/TemporaryItems/NSIRD_screencaptureui_ZfB1tD/Screenshot.png
- If you read a file that exists but has empty contents you will receive a system reminder warning in place of file contents."""
        target = resolve_workspace_path(workspace_root, file_path)
        if not target.exists():
            return f"File does not exist: {file_path}"
        if target.is_dir():
            return f"Path is a directory, not a file: {file_path}"

        raw = target.read_text(encoding="utf-8")
        all_lines = raw.splitlines()

        if not all_lines:
            return "<system-reminder>File is empty.</system-reminder>"

        start = (offset - 1) if offset and offset >= 1 else 0
        end = start + (limit if limit and limit > 0 else 2000)
        selected = all_lines[start:end]

        result_lines: list[str] = []
        for i, line in enumerate(selected, start=start + 1):
            if len(line) > 2000:
                line = line[:2000]
            result_lines.append(f"{i:6d}\t{line}")

        return "\n".join(result_lines)

    @tool
    def Write(file_path: str, content: str) -> str:
        """Writes a file to the local filesystem.

Usage:
- This tool will overwrite the existing file if there is one at the provided path.
- If this is an existing file, you MUST use the Read tool first to read the file's contents. This tool will fail if you did not read the file first.
- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.
- NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
- Only use emojis if the user explicitly requests it. Avoid writing emojis to files unless asked."""
        target = resolve_workspace_path(workspace_root, file_path)
        if mutation_guard is not None:
            denial = mutation_guard(str(target))
            if denial:
                return f"Write blocked: {denial}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to {file_path}"

    @tool
    def LS(path: str, ignore: Optional[list[str]] = None) -> str:
        """Lists files and directories in a given path. The path parameter must be an absolute path, not a relative path. You can optionally provide an array of glob patterns to ignore with the ignore parameter. You should generally prefer the Glob and Grep tools, if you know which directories to search."""
        target = resolve_workspace_path(workspace_root, path)
        if not target.exists():
            return f"Path does not exist: {path}"
        if not target.is_dir():
            return f"Path is not a directory: {path}"

        ignore_patterns = ignore or []
        entries = sorted(
            target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
        )
        if not entries:
            return f"No entries in {path}"

        lines: list[str] = []
        for entry in entries:
            rel = entry.relative_to(workspace_root)
            name = str(rel)
            if any(fnmatch.fnmatch(name, pat) for pat in ignore_patterns):
                continue
            marker = "/" if entry.is_dir() else ""
            lines.append(f"{rel}{marker}")

        if not lines:
            return f"No entries in {path} (all filtered by ignore patterns)"
        return "\n".join(lines)

    return [Read, Write, LS]
