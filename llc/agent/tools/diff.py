import difflib
import os
from pathlib import Path

from langchain.tools import tool
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict

_IGNORED_DIRS = {
    ".git",
    ".cursor",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "node_modules",
}
_MAX_FILE_BYTES = 512_000
_MAX_DIFF_CHARS = 120_000
_SESSION_BASELINES: dict[str, dict[str, "_SnapshotEntry"]] = {}


class _SnapshotEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str
    content: str
    size: int


def _snapshot_entry(path: Path) -> _SnapshotEntry:
    try:
        data = path.read_bytes()
    except OSError as exc:
        return _SnapshotEntry(kind="error", content=str(exc), size=0)
    size = len(data)
    if size > _MAX_FILE_BYTES:
        return _SnapshotEntry(kind="large", content="", size=size)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return _SnapshotEntry(kind="binary", content="", size=size)
    return _SnapshotEntry(kind="text", content=text, size=size)


def _capture_workspace_snapshot(workspace_root: Path) -> dict[str, _SnapshotEntry]:
    snapshot: dict[str, _SnapshotEntry] = {}
    for dirpath, dirnames, filenames in os.walk(workspace_root):
        dirnames[:] = sorted(d for d in dirnames if d not in _IGNORED_DIRS)
        base = Path(dirpath)
        for filename in sorted(filenames):
            full_path = base / filename
            if full_path.is_symlink():
                continue
            rel = full_path.relative_to(workspace_root).as_posix()
            snapshot[rel] = _snapshot_entry(full_path)
    return snapshot


def _entry_label(entry: _SnapshotEntry) -> str:
    if entry.kind == "text":
        return "text"
    if entry.kind == "binary":
        return f"binary ({entry.size} bytes)"
    if entry.kind == "large":
        return f"large ({entry.size} bytes)"
    return f"error ({entry.content})"


def _render_session_diff(
    baseline: dict[str, _SnapshotEntry],
    current: dict[str, _SnapshotEntry],
) -> str:
    lines: list[str] = []
    for rel_path in sorted(set(baseline) | set(current)):
        before = baseline.get(rel_path)
        after = current.get(rel_path)
        if before == after:
            continue

        lines.append(f"diff --git a/{rel_path} b/{rel_path}")

        if before is None and after is not None:
            if after.kind == "text":
                lines.extend(
                    difflib.unified_diff(
                        [],
                        after.content.splitlines(),
                        fromfile=f"a/{rel_path}",
                        tofile=f"b/{rel_path}",
                        lineterm="",
                    )
                )
            else:
                lines.append(f"+[file added: {_entry_label(after)}]")
            continue

        if before is not None and after is None:
            if before.kind == "text":
                lines.extend(
                    difflib.unified_diff(
                        before.content.splitlines(),
                        [],
                        fromfile=f"a/{rel_path}",
                        tofile=f"b/{rel_path}",
                        lineterm="",
                    )
                )
            else:
                lines.append(f"-[file removed: {_entry_label(before)}]")
            continue

        if before is None or after is None:
            continue

        if before.kind == "text" and after.kind == "text":
            lines.extend(
                difflib.unified_diff(
                    before.content.splitlines(),
                    after.content.splitlines(),
                    fromfile=f"a/{rel_path}",
                    tofile=f"b/{rel_path}",
                    lineterm="",
                )
            )
            continue

        lines.append(f"[file changed: {_entry_label(before)} -> {_entry_label(after)}]")

    if not lines:
        return "No changes since session start."

    output = "=== Session changes ===\n" + "\n".join(lines)
    if len(output) > _MAX_DIFF_CHARS:
        return output[:_MAX_DIFF_CHARS] + "\n\n... (diff truncated)"
    return output


def make_diff_tools(workspace_root: Path) -> list[BaseTool]:
    key = str(workspace_root.resolve())
    if key not in _SESSION_BASELINES:
        _SESSION_BASELINES[key] = _capture_workspace_snapshot(workspace_root)

    @tool
    def ShowDiff() -> str:
        """Shows a session diff of workspace changes since the agent session began.

Use this after every successful implementation so the user can see what changed in this session, or when the user asks to see the diff. This works in both git and non-git folders by comparing current workspace state with a snapshot captured at session start."""
        baseline = _SESSION_BASELINES.get(key)
        if baseline is None:
            baseline = _capture_workspace_snapshot(workspace_root)
            _SESSION_BASELINES[key] = baseline
        current = _capture_workspace_snapshot(workspace_root)
        return _render_session_diff(baseline, current)

    metadata = dict(getattr(ShowDiff, "metadata", {}) or {})
    metadata.update(
        {
            "user_facing": True,
            "render_mode": "side_by_side_diff",
        }
    )
    ShowDiff.metadata = metadata

    return [ShowDiff]
