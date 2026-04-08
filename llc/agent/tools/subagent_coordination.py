import json
from pathlib import Path
from typing import Optional

from langchain.tools import tool
from langchain_core.tools import BaseTool

from llc.agent.subagents.coordination import AgentCoordinationLayer
from llc.agent.tools._paths import resolve_workspace_path


def _render(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str)


def make_subagent_coordination_tools(
    coordination: AgentCoordinationLayer,
    *,
    subagent_id: str,
    workspace_root: Path,
) -> list[BaseTool]:
    clean_subagent_id = subagent_id.strip()

    @tool
    def SendMessage(recipient_subagent_id: str, content: str) -> str:
        """Send a direct coordination message to another sub-agent."""
        return _render(
            coordination.send_message(
                sender_id=clean_subagent_id,
                recipient_id=recipient_subagent_id,
                content=content,
            )
        )

    @tool
    def ReadInbox(limit: Optional[int] = None) -> str:
        """Read new inbox messages for this sub-agent."""
        return _render(
            coordination.read_inbox(
                clean_subagent_id,
                limit=limit,
            )
        )

    @tool
    def ReadTeamStatus() -> str:
        """Read a global snapshot of active sub-agents and current tasks."""
        return _render(coordination.read_team_status(agent_id=clean_subagent_id))

    @tool
    def ReadSharedNotes(limit: Optional[int] = None) -> str:
        """Read team-shared notes published by agents."""
        return _render(
            coordination.read_shared_notes(
                agent_id=clean_subagent_id,
                limit=limit,
            )
        )

    @tool
    def AppendSharedNote(note: str) -> str:
        """Append a concise shared note for the sub-agent team."""
        return _render(
            coordination.append_shared_note(
                agent_id=clean_subagent_id,
                note=note,
            )
        )

    @tool
    def RequestLock(file_path: str, lease_seconds: Optional[int] = None) -> str:
        """Request a lease lock for a file before mutating it."""
        target = resolve_workspace_path(workspace_root, file_path)
        return _render(
            coordination.request_lock(
                agent_id=clean_subagent_id,
                file_path=str(target),
                lease_seconds=lease_seconds,
            )
        )

    @tool
    def ReleaseLock(file_path: str, reason: Optional[str] = None) -> str:
        """Release a held file lock when work is complete."""
        target = resolve_workspace_path(workspace_root, file_path)
        return _render(
            coordination.release_lock(
                agent_id=clean_subagent_id,
                file_path=str(target),
                reason=(reason or ""),
            )
        )

    @tool
    def ReviewHeldLocks() -> str:
        """Review currently held lock leases and expiry proximity."""
        return _render(coordination.review_held_locks(clean_subagent_id))

    @tool
    def RespondLockReview(decisions: list[dict]) -> str:
        """Respond to lock review with KEEP or RELEASE per held file lock."""
        normalized: list[dict] = []
        for item in decisions:
            if not isinstance(item, dict):
                continue
            raw_path = str(item.get("file_path", "")).strip()
            if not raw_path:
                continue
            target = resolve_workspace_path(workspace_root, raw_path)
            normalized.append(
                {
                    "file_path": str(target),
                    "action": str(item.get("action", "")),
                }
            )
        return _render(
            coordination.respond_lock_review(
                agent_id=clean_subagent_id,
                decisions=normalized,
            )
        )

    return [
        SendMessage,
        ReadInbox,
        ReadTeamStatus,
        ReadSharedNotes,
        AppendSharedNote,
        RequestLock,
        ReleaseLock,
        ReviewHeldLocks,
        RespondLockReview,
    ]
