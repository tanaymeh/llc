import json
from typing import Optional

from langchain.tools import tool
from langchain_core.tools import BaseTool

from llc.agent.subagents.coordination import SubAgentCoordinationClient


def _render(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str)


def make_subagent_coordination_tools(
    client: SubAgentCoordinationClient,
) -> list[BaseTool]:
    @tool
    def SubmitTaskPlan(steps: list[str] | str) -> str:
        """Submit a mandatory multi-step plan before using any other tool."""
        return _render(client.submit_plan(steps))

    @tool
    def SendPeerMessage(
        to_worker: str,
        message: str,
        kind: Optional[str] = None,
        correlation_id: Optional[str] = None,
        in_reply_to: Optional[str] = None,
        ttl_s: Optional[int] = None,
        task_ref: Optional[str] = None,
    ) -> str:
        """Send a structured message to another sub-agent worker."""
        return _render(
            client.send_message(
                to_worker,
                body=message,
                kind=(kind or "question"),
                correlation_id=(correlation_id or ""),
                in_reply_to=(in_reply_to or ""),
                ttl_s=max(int(ttl_s or 900), 30),
                task_ref=(task_ref or ""),
            )
        )

    @tool
    def HasInboxMessages() -> str:
        """Check whether this worker currently has unread inbox messages."""
        return _render(client.has_unread_messages())

    @tool
    def ReadInbox(limit: Optional[int] = None, only_unread: Optional[bool] = None) -> str:
        """Read inbox messages for this worker."""
        return _render(
            client.read_inbox(
                limit=max(int(limit or 10), 1),
                mark_read=True,
                only_unread=bool(only_unread),
            )
        )

    @tool
    def WaitForPeerMessage(timeout_ms: Optional[int] = None, read_limit: Optional[int] = None) -> str:
        """Wait briefly for peer messages, then return inbox state."""
        wait_payload = client.wait_for_message(timeout_ms=timeout_ms)
        if not wait_payload.get("ok"):
            return _render(wait_payload)
        if not wait_payload.get("has_message"):
            return _render(wait_payload)
        inbox_payload = client.read_inbox(
            limit=max(int(read_limit or 10), 1),
            mark_read=True,
            only_unread=True,
        )
        merged = dict(wait_payload)
        merged["inbox"] = inbox_payload
        return _render(merged)

    @tool
    def PostSharedNote(channel: str, content: str) -> str:
        """Post a short note to the shared worker board."""
        return _render(client.post_note(channel=channel, content=content))

    @tool
    def ReadSharedNotes(channel: Optional[str] = None, limit: Optional[int] = None) -> str:
        """Read notes from the shared worker board."""
        return _render(
            client.read_notes(
                channel=(channel or ""),
                limit=max(int(limit or 20), 1),
            )
        )

    @tool
    def ClaimScope(scope: str, ttl_s: Optional[int] = None, reason: Optional[str] = None) -> str:
        """Claim an exclusive mutable scope (typically a file path or module)."""
        return _render(
            client.claim_scope(
                scope=scope,
                ttl_s=ttl_s,
                reason=(reason or ""),
            )
        )

    @tool
    def ReleaseScope(scope: str) -> str:
        """Release a previously acquired mutable scope claim."""
        return _render(client.release_scope(scope=scope))

    @tool
    def ListMyClaims() -> str:
        """List active scope claims owned by this worker."""
        return _render(client.list_claims())

    @tool
    def GetCoordinationState() -> str:
        """Return current coordination state for this worker."""
        return _render(client.state())

    return [
        SubmitTaskPlan,
        SendPeerMessage,
        HasInboxMessages,
        ReadInbox,
        WaitForPeerMessage,
        PostSharedNote,
        ReadSharedNotes,
        ClaimScope,
        ReleaseScope,
        ListMyClaims,
        GetCoordinationState,
    ]
