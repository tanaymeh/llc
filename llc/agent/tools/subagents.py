import json
from typing import Optional

from langchain.tools import tool
from langchain_core.tools import BaseTool

from llc.agent.subagents import SubAgentRuntime


def _render(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str)


def make_subagent_tools(runtime: SubAgentRuntime) -> list[BaseTool]:
    @tool
    def LaunchSubagent(
        task: str,
        context: Optional[str] = None,
        name: Optional[str] = None,
    ) -> str:
        """Launch a worker sub-agent with an assigned task and optional context."""
        return _render(
            runtime.launch_subagent(
                task,
                context=(context or ""),
                source="orchestrator",
                name=name,
            )
        )

    @tool
    def GetSubagentReport(ids: Optional[list[str]] = None) -> str:
        """Get progress and status reports for running and completed sub-agents."""
        return _render(runtime.get_subagent_report(ids))

    @tool
    def WaitSubagents(ids: Optional[list[str]] = None, timeout_ms: Optional[int] = None) -> str:
        """Wait for selected sub-agents (or all) until completion or timeout."""
        return _render(runtime.wait_subagents(ids, timeout_ms))

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
