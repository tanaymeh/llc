from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from llc.agent.llm import build_chat_model
from llc.agent.llm_retry import ainvoke_with_retry
from llc.agent.message_utils import message_text
from llc.config import Settings
from llc.observability import build_langchain_config, start_child_span, update_observation

from .base import HookContext
from .ordering import HOOK_ORDER_TOOL_OUTPUT_SUMMARY

_TOOL_SUMMARY_TARGETS = frozenset({"Read", "LS", "Glob", "Grep", "code_grep"})
_MAX_TOOL_SUMMARY_INPUT_CHARS = 25000
_MAX_TOOL_SUMMARY_OUTPUT_CHARS = 2000
_TOOL_SUMMARY_PREFIX = "[tool-output-summary]"
_TOOL_SUMMARY_PROMPT = (
    "You summarize coding-tool outputs for chat-history compression.\n"
    "Produce a concise factual summary that preserves details needed for future steps.\n"
    "Focus on key findings, file paths, counts, errors, and actionable outcomes.\n"
    "Do not include markdown headings or preamble.\n"
    "Do not invent details.\n"
    "Keep it short, concise and information dense."
)


class ToolOutputSummaryHook:
    name = "tool_output_summary"
    order = HOOK_ORDER_TOOL_OUTPUT_SUMMARY
    execution_mode = "background"

    async def prepare_after_turn(
        self,
        ctx: HookContext,
    ) -> dict[str, tuple[str, str]] | None:
        candidates = {
            tool_call_id: (tool_name, content)
            for tool_call_id, (tool_name, content) in ctx.tool_output_candidates.items()
            if tool_name in _TOOL_SUMMARY_TARGETS and content.strip()
        }
        if not candidates:
            return None

        summaries: dict[str, tuple[str, str]] = {}
        for tool_call_id, (tool_name, content) in candidates.items():
            summary = await _summarize_tool_output(
                settings=ctx.settings,
                tool_name=tool_name,
                content=content,
            )
            if summary:
                summaries[tool_call_id] = (tool_name, summary)
        return summaries or None

    async def apply_prepared(
        self,
        ctx: HookContext,
        prepared: dict[str, tuple[str, str]],
    ) -> str | None:
        replaced_count = await _swap_tool_output_summaries(
            agent=ctx.agent,
            thread_id=ctx.thread_id,
            summaries=prepared,
        )
        if replaced_count <= 0:
            return None
        noun = "result" if replaced_count == 1 else "results"
        return f"Tool outputs summarized ({replaced_count} {noun} compressed)."


async def _swap_tool_output_summaries(
    *,
    agent: Any,
    thread_id: str,
    summaries: dict[str, tuple[str, str]],
) -> int:
    if not summaries:
        return 0

    state = await agent.aget_state(_graph_config(thread_id))
    values = getattr(state, "values", {})
    messages = values.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return 0

    changed = False
    replaced_count = 0
    replacement: list[Any] = [RemoveMessage(id=REMOVE_ALL_MESSAGES)]
    for message in messages:
        updated = message
        if isinstance(message, ToolMessage):
            tool_call_id = str(getattr(message, "tool_call_id", "") or "")
            summary_payload = summaries.get(tool_call_id)
            if summary_payload is not None:
                raw_content = message_text(message.content, include_reasoning=True).strip()
                if not raw_content.startswith(_TOOL_SUMMARY_PREFIX):
                    tool_name = summary_payload[0] or _tool_name_from_message(message)
                    summary = summary_payload[1]
                    updated = message.model_copy(
                        update={
                            "content": (
                                f"{_TOOL_SUMMARY_PREFIX}\n"
                                f"tool: {tool_name}\n"
                                f"{summary}"
                            )
                        }
                    )
                    changed = True
                    replaced_count += 1
        replacement.append(updated.model_copy(update={"id": None}))

    if not changed:
        return 0

    await agent.aupdate_state(
        _graph_config(thread_id),
        {"messages": replacement},
        as_node="llm",
    )
    return replaced_count


async def _summarize_tool_output(
    *,
    settings: Settings,
    tool_name: str,
    content: str,
) -> str:
    compact_model_name = settings.compact_model_name or settings.model_name
    model = build_chat_model(settings, settings.compact_model_name)

    clipped = content.strip()
    if len(clipped) > _MAX_TOOL_SUMMARY_INPUT_CHARS:
        clipped = clipped[: _MAX_TOOL_SUMMARY_INPUT_CHARS - 3] + "..."

    request = HumanMessage(
        content=(
            f"tool={tool_name}\n"
            "Summarize the following tool output:\n\n"
            f"{clipped}"
        )
    )
    run_config = build_langchain_config(
        tags=("llc", "api", "tool-summary"),
        metadata={
            "llc_model_name": compact_model_name,
            "llc_tool_name": tool_name,
        },
    )
    with start_child_span(
        "llc.api.tool_summary",
        input_payload={"tool_name": tool_name},
        tags=("llc", "api", "tool-summary"),
        metadata={
            "llc_model_name": compact_model_name,
            "llc_tool_name": tool_name,
        },
        as_type="generation",
        model_name=compact_model_name,
    ) as summary_span:
        response = await ainvoke_with_retry(
            model,
            [SystemMessage(content=_TOOL_SUMMARY_PROMPT), request],
            config=run_config,
        )

    summary = message_text(response.content, include_reasoning=True).strip()
    if len(summary) > _MAX_TOOL_SUMMARY_OUTPUT_CHARS:
        summary = summary[: _MAX_TOOL_SUMMARY_OUTPUT_CHARS - 3] + "..."
    if not summary:
        update_observation(
            summary_span,
            output={"status": "empty_summary", "tool_name": tool_name},
        )
        return ""

    update_observation(
        summary_span,
        output={
            "status": "ok",
            "tool_name": tool_name,
            "summary_preview": _preview_text(summary, 220),
        },
    )
    return summary


def _graph_config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


def _tool_name_from_message(message: ToolMessage) -> str:
    kwargs = getattr(message, "additional_kwargs", {}) or {}
    raw_name = kwargs.get("tool_name", "tool")
    return str(raw_name or "tool")


def _preview_text(text: str, limit: int) -> str:
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."
