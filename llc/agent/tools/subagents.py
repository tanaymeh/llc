import json
from typing import Optional

from langchain.tools import tool
from langchain_core.tools import BaseTool

from llc.agent.subagents import SubAgentRuntime


def _render(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str)


def _normalize_ids(ids: Optional[list[str] | str]) -> list[str] | None:
    if ids is None:
        return None
    if isinstance(ids, list):
        normalized = [str(item).strip() for item in ids if str(item).strip()]
        return normalized or None
    raw = ids.strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = None
    if isinstance(parsed, list):
        normalized = [str(item).strip() for item in parsed if str(item).strip()]
        return normalized or None
    if "," in raw:
        normalized = [part.strip() for part in raw.split(",") if part.strip()]
        return normalized or None
    return [raw]


def make_subagent_tools(runtime: SubAgentRuntime) -> list[BaseTool]:
    @tool
    def LaunchSubagent(
        task: str,
        name: str,
        context: Optional[str] = None,
    ) -> str:
        """Launch a worker sub-agent with an assigned task, required name, and optional context."""
        return _render(
            runtime.launch_subagent(
                task,
                context=(context or ""),
                source="orchestrator",
                name=name,
            )
        )

    @tool
    def GetSubagentReport(ids: Optional[list[str] | str] = None) -> str:
        """Get progress and status reports for running and completed sub-agents."""
        return _render(runtime.get_subagent_report(_normalize_ids(ids)))

    @tool
    def WaitSubagents(
        ids: Optional[list[str] | str] = None,
        timeout_ms: Optional[int] = None,
    ) -> str:
        """Wait briefly for selected sub-agents (or all); pass timeout_ms=0 for unbounded wait."""
        return _render(runtime.wait_subagents(_normalize_ids(ids), timeout_ms))

    @tool
    def ReviseSubagent(subagent_id: str, feedback: str) -> str:
        """Request revisions from a sub-agent using explicit feedback."""
        return _render(
            runtime.revise_subagent(
                subagent_id,
                feedback,
                source="orchestrator",
            )
        )

    @tool
    def InterruptSubagent(subagent_id: str, instruction: str) -> str:
        """Interrupt and redirect an in-flight sub-agent with new instructions."""
        return _render(
            runtime.interrupt_subagent(
                subagent_id,
                instruction,
                source="orchestrator",
            )
        )

    @tool
    def TerminateSubagent(subagent_id: str, reason: Optional[str] = None) -> str:
        """Terminate a running sub-agent or mark a completed one as terminated."""
        return _render(
            runtime.terminate_subagent(
                subagent_id,
                reason=reason or "",
            )
        )

    return [
        LaunchSubagent,
        GetSubagentReport,
        WaitSubagents,
        ReviseSubagent,
        InterruptSubagent,
        TerminateSubagent,
    ]
