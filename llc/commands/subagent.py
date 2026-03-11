from __future__ import annotations

from langchain_core.messages import HumanMessage

from llc.agent.compact import aget_history_messages
from llc.agent.message_utils import message_text
from llc.commands import Command, CommandResult, ReplContext

_CONTEXT_CHAR_LIMIT = 5000
_BLOCKED_CONTEXT_TOKENS = (
    "sub-agent",
    "subagent",
    "/subagent",
    "launchsubagent",
    "revise subagent",
    "interrupt subagent",
    "terminate subagent",
    "orchestrator",
)


class SubagentCommand(Command):
    @property
    def name(self) -> str:
        return "/subagent"

    @property
    def description(self) -> str:
        return "Spawn a manual worker with `/subagent {TASK}`"

    async def execute(self, args: str, ctx: ReplContext) -> CommandResult:
        if not ctx.settings.sub_agent_mode_enabled:
            return CommandResult(
                message="`sub-agent-mode` is disabled. Run `/enable sub-agent-mode` first."
            )
        if ctx.subagent_runtime is None:
            return CommandResult(message="Sub-agent runtime is unavailable.")

        task, error = _parse_braced_task(args)
        if error:
            return CommandResult(message=error)

        history = await aget_history_messages(ctx.agent, ctx.thread_id)
        context = _build_context(history, ctx.settings.sub_agent_context_messages)
        result = ctx.subagent_runtime.launch_subagent(
            task,
            context=context,
            source="manual-command",
        )
        if not result.get("ok"):
            return CommandResult(message=str(result.get("error", "Failed to launch sub-agent.")))

        subagent_id = str(result.get("id", "unknown"))
        status = str(result.get("status", "running"))
        active = ctx.subagent_runtime.get_subagent_report().get("active_count", 0)
        max_workers = ctx.settings.max_sub_agents
        return CommandResult(
            message=(
                f"Spawned `{subagent_id}` (`{status}`). "
                f"Active workers: {active}/{max_workers}."
            ),
            data={"spawned_subagent_id": subagent_id},
        )


def _parse_braced_task(raw: str) -> tuple[str, str | None]:
    value = raw.strip()
    if not value:
        return "", "Usage: `/subagent {SUB_AGENT_TASK}`"
    if not value.startswith("{") or not value.endswith("}"):
        return "", "Task must be wrapped in braces. Usage: `/subagent {SUB_AGENT_TASK}`"
    inner = value[1:-1]
    if "{" in inner or "}" in inner:
        return "", "Malformed braces. Usage: `/subagent {SUB_AGENT_TASK}`"
    task = inner.strip()
    if not task:
        return "", "Sub-agent task cannot be empty. Usage: `/subagent {SUB_AGENT_TASK}`"
    return task, None


def _build_context(messages: list, limit_messages: int) -> str:
    human_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
    selected = human_messages[-limit_messages:] if limit_messages > 0 else human_messages
    lines: list[str] = []
    for msg in selected:
        text = _sanitize_context_text(
            message_text(getattr(msg, "content", "")).strip()
        )
        if not text:
            continue
        if len(text) > 700:
            text = text[:700] + "..."
        lines.append(f"User: {text}")
    context = "\n".join(lines).strip()
    if len(context) > _CONTEXT_CHAR_LIMIT:
        context = context[-_CONTEXT_CHAR_LIMIT:]
    return context


def _sanitize_context_text(text: str) -> str:
    kept_lines: list[str] = []
    for line in text.splitlines():
        lowered = line.lower()
        if any(token in lowered for token in _BLOCKED_CONTEXT_TOKENS):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines).strip()
