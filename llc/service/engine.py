from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from llc.agent import build_agent_graph
from llc.agent.hooks import (
    AutoCompactHook,
    Hook,
    HookContext,
    TokenCounterHook,
    ToolOutputSummaryHook,
    hook_order,
)
from llc.agent.subagents import SubAgentRuntime
from llc.commands import CommandRegistry, ReplContext
from llc.config import Settings
from llc.models import AvailableModel, fetch_models, get_model_pricing
from llc.observability import (
    merge_langchain_config,
    start_api_turn_trace,
    update_observation,
)
from llc.service.events import (
    CommandOutput,
    ErrorOccurred,
    Event,
    SessionRestored,
    SubagentStatusUpdate,
    TextDelta,
    ToolResultEvent,
    TurnCompleted,
    TurnStarted,
    UsageUpdate,
)
from llc.service.prompt_registry import PromptRegistry
from llc.service.stream_adapter import StreamAdapter

_SESSION_INTERRUPT_MESSAGE = "Session Interrupted, what should be done differently?"
_MANUAL_SUBAGENT_FOLLOWUP_PROMPT = (
    "A manual /subagent launch just occurred.\n"
    "Continue orchestration for all currently active workers.\n"
    "Do not launch new workers unless explicitly required for safety.\n"
    "Keep this response open until all workers complete.\n"
    "Provide concise progress updates and then a final combined outcome."
)
_MAX_TRACE_TEXT_PREVIEW_CHARS = 500


def _preview_text(text: str, limit: int = _MAX_TRACE_TEXT_PREVIEW_CHARS) -> str:
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


class SessionEngine:
    def __init__(
        self,
        settings: Settings,
        registry: CommandRegistry,
        *,
        session_id: str | None = None,
        store: Any | None = None,
        enable_langfuse_tracing: bool = False,
    ) -> None:
        self._settings = settings
        self._registry = registry
        self._store = store
        self._session_id = session_id or f"session-{uuid.uuid4().hex[:12]}"
        self._thread_id = f"service-{uuid.uuid4().hex[:12]}"
        self._prompt_registry = PromptRegistry(settings.prompts_dir)
        self._subagent_runtime: SubAgentRuntime | None = None
        self._agent: Any | None = None
        self._token_hook = TokenCounterHook()
        self._hooks: list[Hook] = sorted(
            [self._token_hook, ToolOutputSummaryHook(), AutoCompactHook()],
            key=hook_order,
        )
        self._available_models: list[AvailableModel] = []
        self._initialized = False
        self._turn_lock: asyncio.Lock | None = None
        self._langfuse_enabled = enable_langfuse_tracing

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def model_name(self) -> str:
        return self._settings.model_name

    @property
    def sub_agent_mode_enabled(self) -> bool:
        return self._settings.sub_agent_mode_enabled

    @property
    def available_models(self) -> list[AvailableModel]:
        return list(self._available_models)

    def get_subagent_report(self, *, include_all: bool = False) -> dict[str, Any]:
        runtime = self._subagent_runtime
        if runtime is None:
            return {
                "ok": True,
                "active_count": 0,
                "max_sub_agents": self._settings.max_sub_agents,
                "workers": [],
            }
        return runtime.get_subagent_report(include_all=include_all)

    async def initialize(self) -> None:
        if self._initialized:
            return

        self._turn_lock = asyncio.Lock()
        self._subagent_runtime = SubAgentRuntime(
            self._settings,
            build_subagent=lambda settings: build_agent_graph(
                settings,
                role="subagent",
                prompt_registry=self._prompt_registry,
            ),
            prompt_registry=self._prompt_registry,
        )
        role = "orchestrator" if self._settings.sub_agent_mode_enabled else "default"
        self._agent = build_agent_graph(
            self._settings,
            role=role,
            subagent_runtime=self._subagent_runtime,
            prompt_registry=self._prompt_registry,
        )

        self._available_models = await fetch_models(
            base_url=self._settings.openai_base_url,
            api_key=self._settings.openai_api_key,
        )

        if self._store is not None:
            from llc.storage.models import SessionRecord

            await self._store.initialize()
            existing = await self._store.get_session(self._session_id)
            now = time.time()
            if existing is None:
                await self._store.create_session(
                    SessionRecord(
                        id=self._session_id,
                        model_name=self._settings.model_name,
                        sub_agent_mode=self._settings.sub_agent_mode_enabled,
                        created_at=now,
                        updated_at=now,
                    )
                )
        self._initialized = True

    async def shutdown(self) -> None:
        if self._subagent_runtime is not None:
            self._subagent_runtime.shutdown()
        if self._store is not None:
            await self._store.close()
        self._initialized = False

    async def interrupt(self) -> AsyncIterator[Event]:
        await self.initialize()
        self._terminate_active_subagents(reason="session_interrupted")
        message = self._safe_prompt(
            "interrupt",
            key="session_interrupt_message",
            fallback=_SESSION_INTERRUPT_MESSAGE,
        )
        event = CommandOutput(message=message)
        await self._persist_event(event, turn_id="")
        yield event

    async def send_message(self, text: str) -> AsyncIterator[Event]:
        await self.initialize()
        if self._turn_lock is None:
            raise RuntimeError("Session engine lock was not initialized")
        if self._agent is None:
            raise RuntimeError("Agent was not initialized")

        async with self._turn_lock:
            turn_id = f"turn-{uuid.uuid4().hex[:10]}"
            started = TurnStarted(turn_id=turn_id)
            await self._persist_event(started, turn_id=turn_id)
            yield started
            await self._persist_message(role="user", content=text)

            match = self._registry.match(text)
            if match is not None:
                command, args = match
                async for event in self._run_command(command, args, turn_id):
                    yield event
                return

            async for event in self._stream_agent_response(
                prompt_text=text,
                turn_id=turn_id,
            ):
                yield event

    async def list_sessions(self) -> list[dict[str, Any]]:
        await self.initialize()
        if self._store is None:
            return []
        records = await self._store.list_sessions()
        return [record.model_dump() for record in records]

    async def restore_session(self, session_id: str) -> AsyncIterator[Event]:
        await self.initialize()
        if self._store is None:
            event = ErrorOccurred(message="Session store is not configured.")
            await self._persist_event(event, turn_id="")
            yield event
            return

        messages = await self._store.get_messages(session_id)
        count = len(messages)
        if count == 0:
            event = ErrorOccurred(message=f"No session data found for `{session_id}`.")
            await self._persist_event(event, turn_id="")
            yield event
            return

        await self._restore_messages_to_agent(messages)
        self._session_id = session_id
        event = SessionRestored(session_id=session_id, message_count=count)
        await self._persist_event(event, turn_id="")
        yield event

    async def _run_command(
        self,
        command: Any,
        args: str,
        turn_id: str,
    ) -> AsyncIterator[Event]:
        if self._agent is None:
            error = ErrorOccurred(message="Agent is unavailable.")
            await self._persist_event(error, turn_id=turn_id)
            yield error
            return

        context = ReplContext(
            settings=self._settings,
            agent=self._agent,
            thread_id=self._thread_id,
            subagent_runtime=self._subagent_runtime,
            prompt_registry=self._prompt_registry,
        )
        result = await command.execute(args, context)
        self._settings = context.settings
        self._agent = context.agent
        self._subagent_runtime = context.subagent_runtime

        if self._store is not None:
            await self._store.update_session(
                self._session_id,
                model_name=self._settings.model_name,
                sub_agent_mode=self._settings.sub_agent_mode_enabled,
                updated_at=time.time(),
            )

        event = CommandOutput(
            message=result.message,
            should_exit=result.should_exit,
            data=result.data,
        )
        await self._persist_event(event, turn_id=turn_id)
        yield event

        if result.message:
            await self._persist_message(role="assistant", content=result.message)

        spawned_subagent_id = str(result.data.get("spawned_subagent_id", "")).strip()
        if (
            spawned_subagent_id
            and self._settings.sub_agent_mode_enabled
            and self._subagent_runtime is not None
        ):
            report = self._subagent_runtime.get_subagent_report()
            try:
                active_count = int(report.get("active_count", 0) or 0)
            except Exception:
                active_count = 0
            if active_count > 0:
                followup_prompt = self._safe_prompt(
                    "interrupt",
                    key="manual_subagent_followup_prompt",
                    fallback=_MANUAL_SUBAGENT_FOLLOWUP_PROMPT,
                )
                async for followup_event in self._stream_agent_response(
                    prompt_text=followup_prompt,
                    turn_id=turn_id,
                    persist_user=False,
                ):
                    yield followup_event
                return

        completed = TurnCompleted(input_tokens=0, output_tokens=0)
        await self._persist_event(completed, turn_id=turn_id)
        yield completed

    async def _stream_agent_response(
        self,
        *,
        prompt_text: str,
        turn_id: str,
        persist_user: bool = False,
    ) -> AsyncIterator[Event]:
        if self._agent is None:
            error = ErrorOccurred(message="Agent is unavailable.")
            await self._persist_event(error, turn_id=turn_id)
            yield error
            return

        if persist_user:
            await self._persist_message(role="user", content=prompt_text)

        adapter = StreamAdapter()
        assistant_parts: list[str] = []
        saw_text = False
        last_subagent_snapshot = ""
        tool_output_candidates: dict[str, tuple[str, str]] = {}
        with start_api_turn_trace(
            enabled=self._langfuse_enabled,
            session_id=self._session_id,
            turn_id=turn_id,
            thread_id=self._thread_id,
            model_name=self._settings.model_name,
            sub_agent_mode_enabled=self._settings.sub_agent_mode_enabled,
            prompt_text=prompt_text,
        ) as trace_scope:
            trace_stream_config = dict(trace_scope.langchain_config)
            trace_stream_config.pop("callbacks", None)
            stream_config = merge_langchain_config(
                {"configurable": {"thread_id": self._thread_id}},
                trace_stream_config,
            )
            try:
                async for mode, chunk in self._agent.astream(
                    {"messages": [HumanMessage(content=prompt_text)]},
                    config=stream_config,
                    stream_mode=["messages", "updates"],
                ):
                    events = adapter.parse(mode, chunk)
                    for event in events:
                        if isinstance(event, TextDelta):
                            saw_text = True
                            assistant_parts.append(event.text)
                        elif isinstance(event, ToolResultEvent):
                            tool_call_id = event.tool_call_id.strip()
                            if tool_call_id:
                                tool_output_candidates[tool_call_id] = (
                                    event.tool_name.strip(),
                                    event.content,
                                )
                        await self._persist_event(event, turn_id=turn_id)
                        yield event

                    if adapter.consume_usage_updated():
                        usage_update = self._build_usage_event(
                            turn_input_tokens=adapter.turn_input_tokens,
                            turn_output_tokens=adapter.turn_output_tokens,
                            include_pending_turn=True,
                        )
                        await self._persist_event(usage_update, turn_id=turn_id)
                        yield usage_update

                    subagent_event, snapshot_key = self._build_subagent_status_event(
                        previous_snapshot=last_subagent_snapshot
                    )
                    if subagent_event is not None and snapshot_key is not None:
                        last_subagent_snapshot = snapshot_key
                        await self._persist_event(subagent_event, turn_id=turn_id)
                        yield subagent_event
            except Exception as exc:  # noqa: BLE001
                update_observation(
                    trace_scope.observation,
                    output={"status": "error", "error": str(exc)},
                )
                error = ErrorOccurred(message=str(exc))
                await self._persist_event(error, turn_id=turn_id)
                yield error
                return

            fallback = adapter.fallback_text.strip()
            if not saw_text and fallback:
                text_event = TextDelta(text=fallback)
                assistant_parts.append(fallback)
                await self._persist_event(text_event, turn_id=turn_id)
                yield text_event

            turn_input = adapter.turn_input_tokens
            turn_output = adapter.turn_output_tokens
            if turn_input == 0 and turn_output == 0 and adapter.streamed_text_chars > 0:
                turn_output = max(1, adapter.streamed_text_chars // 4)

            hook_message = await self._run_hooks(
                turn_input_tokens=turn_input,
                turn_output_tokens=turn_output,
                tool_output_candidates=tool_output_candidates,
            )
            usage_final = self._build_usage_event(
                turn_input_tokens=turn_input,
                turn_output_tokens=turn_output,
                include_pending_turn=False,
            )
            await self._persist_event(usage_final, turn_id=turn_id)
            yield usage_final

            final_text = "".join(assistant_parts).strip()
            if final_text:
                await self._persist_message(role="assistant", content=final_text)

            await self._persist_usage(
                input_tokens=turn_input,
                output_tokens=turn_output,
            )
            completed = TurnCompleted(
                input_tokens=turn_input,
                output_tokens=turn_output,
                compact_message=hook_message,
            )
            await self._persist_event(completed, turn_id=turn_id)
            yield completed
            update_observation(
                trace_scope.observation,
                output={
                    "status": "ok",
                    "output_preview": _preview_text(final_text),
                    "turn_input_tokens": turn_input,
                    "turn_output_tokens": turn_output,
                },
            )

    async def _run_hooks(
        self,
        *,
        turn_input_tokens: int,
        turn_output_tokens: int,
        tool_output_candidates: dict[str, tuple[str, str]],
    ) -> str | None:
        if self._agent is None:
            return None
        if (
            turn_input_tokens <= 0
            and turn_output_tokens <= 0
            and not tool_output_candidates
        ):
            return None

        hook_ctx = HookContext(
            agent=self._agent,
            thread_id=self._thread_id,
            settings=self._settings,
            last_turn_input_tokens=turn_input_tokens,
            last_turn_output_tokens=turn_output_tokens,
            available_models=self._available_models,
            compact_prompt=self._settings.compact_prompt,
            tool_output_candidates=tool_output_candidates,
        )
        messages: list[str] = []
        for hook in sorted(self._hooks, key=hook_order):
            try:
                maybe_message = await hook.after_turn(hook_ctx)
            except Exception as exc:  # noqa: BLE001
                maybe_message = f"Hook failed: `{exc!s}`"
            if maybe_message:
                messages.append(maybe_message)
        return "\n".join(messages) if messages else None

    def _build_subagent_status_event(
        self,
        *,
        previous_snapshot: str,
    ) -> tuple[SubagentStatusUpdate | None, str | None]:
        runtime = self._subagent_runtime
        if runtime is None:
            return None, None
        try:
            report = runtime.get_subagent_report()
        except Exception:
            return None, None

        workers = report.get("workers", [])
        if not isinstance(workers, list):
            workers = []
        snapshot_key = json.dumps(
            [
                (
                    str(worker.get("id", "")),
                    str(worker.get("status", "")),
                    float(worker.get("updated_at", 0.0) or 0.0),
                )
                for worker in workers
                if isinstance(worker, dict)
            ],
            sort_keys=True,
            default=str,
        )
        if snapshot_key == previous_snapshot:
            return None, None

        try:
            active_count = int(report.get("active_count", 0) or 0)
        except Exception:
            active_count = 0
        try:
            max_sub_agents = int(report.get("max_sub_agents", 0) or 0)
        except Exception:
            max_sub_agents = 0
        event = SubagentStatusUpdate(
            workers=[worker for worker in workers if isinstance(worker, dict)],
            active_count=max(active_count, 0),
            max_sub_agents=max(max_sub_agents, 0),
        )
        return event, snapshot_key

    def _build_usage_event(
        self,
        *,
        turn_input_tokens: int,
        turn_output_tokens: int,
        include_pending_turn: bool,
    ) -> UsageUpdate:
        subagent_input, subagent_output, subagent_cost = self._subagent_usage_snapshot()
        if include_pending_turn:
            session_input = self._token_hook.session_input + turn_input_tokens + subagent_input
            session_output = (
                self._token_hook.session_output + turn_output_tokens + subagent_output
            )
            session_cost = (
                self._token_hook.session_cost
                + self._turn_cost(turn_input_tokens, turn_output_tokens)
                + subagent_cost
            )
        else:
            session_input = self._token_hook.session_input + subagent_input
            session_output = self._token_hook.session_output + subagent_output
            session_cost = self._token_hook.session_cost + subagent_cost
        return UsageUpdate(
            turn_input_tokens=max(turn_input_tokens, 0),
            turn_output_tokens=max(turn_output_tokens, 0),
            session_input_tokens=max(session_input, 0),
            session_output_tokens=max(session_output, 0),
            session_cost=max(session_cost, 0.0),
        )

    def _turn_cost(self, input_tokens: int, output_tokens: int) -> float:
        pricing = get_model_pricing(self._available_models, self._settings.model_name)
        if pricing is None:
            return 0.0
        return input_tokens * pricing[0] + output_tokens * pricing[1]

    def _subagent_usage_snapshot(self) -> tuple[int, int, float]:
        runtime = self._subagent_runtime
        if runtime is None:
            return 0, 0, 0.0
        try:
            usage = runtime.get_usage_totals()
        except Exception:
            return 0, 0, 0.0

        input_tokens = max(int(usage.get("input_tokens", 0) or 0), 0)
        output_tokens = max(int(usage.get("output_tokens", 0) or 0), 0)
        by_model = usage.get("by_model", {})
        if not isinstance(by_model, dict):
            return input_tokens, output_tokens, 0.0

        cost = 0.0
        for raw_model_name, raw_bucket in by_model.items():
            model_name = str(raw_model_name).strip()
            if not model_name or not isinstance(raw_bucket, dict):
                continue
            model_input = max(int(raw_bucket.get("input_tokens", 0) or 0), 0)
            model_output = max(int(raw_bucket.get("output_tokens", 0) or 0), 0)
            if model_input == 0 and model_output == 0:
                continue
            pricing = get_model_pricing(self._available_models, model_name)
            if pricing is None:
                continue
            cost += model_input * pricing[0] + model_output * pricing[1]
        return input_tokens, output_tokens, cost

    def _terminate_active_subagents(self, *, reason: str) -> None:
        runtime = self._subagent_runtime
        if runtime is None:
            return
        try:
            report = runtime.get_subagent_report()
        except Exception:
            return
        workers = report.get("workers", [])
        if not isinstance(workers, list):
            return
        for worker in workers:
            if not isinstance(worker, dict):
                continue
            status = str(worker.get("status", ""))
            if status not in {"running", "restarting", "terminating"}:
                continue
            subagent_id = str(worker.get("id", "")).strip()
            if not subagent_id:
                continue
            try:
                runtime.terminate_subagent(subagent_id, reason=reason)
            except Exception:
                continue

    def _safe_prompt(self, name: str, *, key: str, fallback: str) -> str:
        try:
            return self._prompt_registry.get(name, key=key)
        except Exception:
            return fallback

    async def _persist_message(
        self,
        *,
        role: str,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
        usage_input_tokens: int = 0,
        usage_output_tokens: int = 0,
    ) -> None:
        if self._store is None:
            return
        from llc.storage.models import MessageRecord

        await self._store.save_message(
            MessageRecord(
                id=f"msg-{uuid.uuid4().hex}",
                session_id=self._session_id,
                role=role,
                content=content,
                tool_calls_json=json.dumps(tool_calls or [], sort_keys=True, default=str),
                usage_input_tokens=max(int(usage_input_tokens), 0),
                usage_output_tokens=max(int(usage_output_tokens), 0),
                created_at=time.time(),
            )
        )
        await self._store.update_session(
            self._session_id,
            model_name=self._settings.model_name,
            sub_agent_mode=self._settings.sub_agent_mode_enabled,
            updated_at=time.time(),
        )

    async def _persist_usage(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        if self._store is None:
            return
        if input_tokens <= 0 and output_tokens <= 0:
            return
        from llc.storage.models import TokenUsageRecord

        await self._store.record_usage(
            TokenUsageRecord(
                session_id=self._session_id,
                model_name=self._settings.model_name,
                input_tokens=max(input_tokens, 0),
                output_tokens=max(output_tokens, 0),
                cost=max(self._turn_cost(input_tokens, output_tokens), 0.0),
                recorded_at=time.time(),
            )
        )

    async def _persist_event(self, event: Event, *, turn_id: str) -> None:
        if self._store is None:
            return
        from llc.storage.models import EventRecord

        await self._store.save_event(
            EventRecord(
                session_id=self._session_id,
                turn_id=turn_id,
                event_type=event.type,
                payload_json=event.model_dump_json(),
                created_at=time.time(),
            )
        )

    async def _restore_messages_to_agent(self, records: list[Any]) -> None:
        if self._agent is None:
            return

        restored: list[Any] = []
        for record in records:
            role = str(getattr(record, "role", "")).strip().lower()
            content = str(getattr(record, "content", ""))
            if role == "user":
                restored.append(HumanMessage(content=content))
                continue
            if role == "assistant":
                raw_tool_calls = getattr(record, "tool_calls_json", "[]")
                try:
                    parsed = json.loads(raw_tool_calls)
                except Exception:
                    parsed = []
                tool_calls = parsed if isinstance(parsed, list) else []
                restored.append(
                    AIMessage(
                        content=content,
                        tool_calls=tool_calls,
                    )
                )
                continue
            if role == "system":
                restored.append(SystemMessage(content=content))
                continue
            if role == "tool":
                restored.append(
                    ToolMessage(
                        content=content,
                        tool_call_id=f"restored-{uuid.uuid4().hex[:8]}",
                    )
                )

        if not restored:
            return

        await self._agent.aupdate_state(
            {"configurable": {"thread_id": self._thread_id}},
            {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *restored]},
            as_node="llm",
        )

