from pathlib import Path
from typing import Optional

from langchain.tools import tool
from langchain_core.tools import BaseTool


def _resolve_path(workspace_root: Path, file_path: str) -> Path:
    candidate = (workspace_root / file_path).resolve()
    try:
        candidate.relative_to(workspace_root)
    except ValueError:
        raise ValueError("Path is outside the workspace root and is not allowed.")
    return candidate


def _apply_edit(content: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    if old_string == new_string:
        raise ValueError("old_string and new_string must be different.")

    if not old_string:
        return new_string + content

    count = content.count(old_string)
    if count == 0:
        raise ValueError(f"old_string not found in file. Make sure it matches exactly, including whitespace.")

    if not replace_all and count > 1:
        raise ValueError(
            f"old_string appears {count} times in the file. "
            "Provide a larger string with more surrounding context to make it unique, "
            "or use replace_all=true to replace every occurrence."
        )

    if replace_all:
        return content.replace(old_string, new_string)
    return content.replace(old_string, new_string, 1)


def make_edit_tools(workspace_root: Path) -> list[BaseTool]:
    @tool
    def Edit(file_path: str, old_string: str, new_string: str, replace_all: Optional[bool] = None) -> str:
        """Performs exact string replacements in files.

Usage:
- You must use your `Read` tool at least once in the conversation before editing. This tool will error if you attempt an edit without reading the file.
- When editing text from Read tool output, ensure you preserve the exact indentation (tabs/spaces) as it appears AFTER the line number prefix. The line number prefix format is: spaces + line number + tab. Everything after that tab is the actual file content to match. Never include any part of the line number prefix in the old_string or new_string.
- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.
- Only use emojis if the user explicitly requests it. Avoid adding emojis to files unless asked.
- The edit will FAIL if `old_string` is not unique in the file. Either provide a larger string with more surrounding context to make it unique or use `replace_all` to change every instance of `old_string`.
- Use `replace_all` for replacing and renaming strings across the file. This parameter is useful if you want to rename a variable for instance."""
        target = _resolve_path(workspace_root, file_path)

        if not target.exists():
            if old_string == "":
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(new_string, encoding="utf-8")
                return f"Created new file {file_path}"
            return f"File does not exist: {file_path}"

        content = target.read_text(encoding="utf-8")
        try:
            updated = _apply_edit(content, old_string, new_string, replace_all or False)
        except ValueError as exc:
            return f"Edit failed: {exc}"

        target.write_text(updated, encoding="utf-8")
        return f"Successfully edited {file_path}"

    @tool
    def MultiEdit(file_path: str, edits: list[dict]) -> str:
        """This is a tool for making multiple edits to a single file in one operation. It is built on top of the Edit tool and allows you to perform multiple find-and-replace operations efficiently. Prefer this tool over the Edit tool when you need to make multiple edits to the same file.

Before using this tool:

1. Use the Read tool to understand the file's contents and context
2. Verify the directory path is correct

To make multiple file edits, provide the following:
1. file_path: The absolute path to the file to modify (must be absolute, not relative)
2. edits: An array of edit operations to perform, where each edit contains:
   - old_string: The text to replace (must match the file contents exactly, including all whitespace and indentation)
   - new_string: The edited text to replace the old_string
   - replace_all: Replace all occurences of old_string. This parameter is optional and defaults to false.

IMPORTANT:
- All edits are applied in sequence, in the order they are provided
- Each edit operates on the result of the previous edit
- All edits must be valid for the operation to succeed - if any edit fails, none will be applied
- This tool is ideal when you need to make several changes to different parts of the same file

CRITICAL REQUIREMENTS:
1. All edits follow the same requirements as the single Edit tool
2. The edits are atomic - either all succeed or none are applied
3. Plan your edits carefully to avoid conflicts between sequential operations"""
        target = _resolve_path(workspace_root, file_path)

        if not target.exists():
            return f"File does not exist: {file_path}"

        content = target.read_text(encoding="utf-8")
        working = content

        for i, edit in enumerate(edits):
            old_s = edit.get("old_string", "")
            new_s = edit.get("new_string", "")
            repl_all = edit.get("replace_all", False)
            try:
                working = _apply_edit(working, old_s, new_s, repl_all)
            except ValueError as exc:
                return f"MultiEdit failed at edit {i + 1}: {exc}. No changes were applied."

        target.write_text(working, encoding="utf-8")
        return f"Successfully applied {len(edits)} edits to {file_path}"

    return [Edit, MultiEdit]
