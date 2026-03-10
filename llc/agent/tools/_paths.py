from pathlib import Path


def resolve_workspace_path(workspace_root: Path, file_path: str) -> Path:
    candidate = (workspace_root / file_path).resolve()
    try:
        candidate.relative_to(workspace_root)
    except ValueError:
        raise ValueError("Path is outside the workspace root and is not allowed.")
    return candidate
