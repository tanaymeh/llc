from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Any

from llc.agent import build_agent_graph
from llc.agent.hooks import (
    AutoCompactHook,
    Hook,
    HookContext,
    HookExecutionMode,
    TokenCounterHook,
    ToolOutputSummaryHook,
    hook_order,
)
from llc.agent.subagents import SubAgentRuntime
from llc.commands import CommandRegistry, ReplContext
from llc.config import Settings
from llc.models import AvailableModel, fetch_models
from llc.service import engine_hooks, engine_persistence, engine_stream
from llc.service.events import (
    CommandOutput,
    ErrorOccurred,
    Event,
    HookStage,
    SessionRestored,
    SubagentStatusUpdate,
    TurnCompleted,
    TurnStarted,
    UsageUpdate,
)
from llc.service.prompt_registry import PromptRegistry

_SESSION_INTERRUPT_MESSAGE = "Session Interrupted, what should be done differently?"
_MANUAL_SUBAGENT_FOLLOWUP_PROMPT = (
    "A manual /subagent launch just occurred.\n"
    "Continue orchestration for all currently active workers.\n"
    "Do not launch new workers unless explicitly required for safety.\n"
    "Keep this response open until all workers complete.\n"
    "Provide concise progress updates and then a final combined outcome."
)


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
        self._blocking_hooks: list[Hook] = [
            hook for hook in self._hooks if self._hook_execution_mode(hook) == "blocking"
        ]
        self._background_hooks: list[Hook] = [
            hook for hook in self._hooks if self._hook_execution_mode(hook) == "background"
        ]
        self._available_models: list[AvailableModel] = []
        self._initialized = False
        self._turn_lock: asyncio.Lock | None = None
        self._hook_apply_lock = asyncio.Lock()
        self._langfuse_enabled = enable_langfuse_tracing
        self._active_turn_task: asyncio.Task[Any] | None = None
        self._active_turn_id: str = ""
        self._interrupt_requested = False
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._hook_runner_tasks: dict[str, asyncio.Task[Any]] = {}
        self._hook_pending_jobs: dict[str, engine_hooks.HookJob] = {}
        self._hook_ready_results: dict[str, engine_hooks.HookPreparedResult] = {}
        self._hook_apply_task: asyncio.Task[Any] | None = None
        self._async_event_subscribers: set[asyncio.Queue[Event]] = set()

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

    def subscribe_async_events(self) -> asyncio.Queue[Event]:
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=200)
        self._async_event_subscribers.add(queue)
        return queue

    def unsubscribe_async_events(self, queue: asyncio.Queue[Event]) -> None:
        self._async_event_subscribers.discard(queue)

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
        background_tasks = list(self._background_tasks)
        for task in background_tasks:
            task.cancel()
        for task in background_tasks:
            with suppress(asyncio.CancelledError):
                await task
        self._background_tasks.clear()
        self._hook_runner_tasks.clear()
        self._hook_pending_jobs.clear()
        self._hook_ready_results.clear()
        self._hook_apply_task = None
        self._async_event_subscribers.clear()
        if self._subagent_runtime is not None:
            self._subagent_runtime.shutdown()
        if self._store is not None:
            await self._store.close()
        self._initialized = False

    async def interrupt(self) -> AsyncIterator[Event]:
        await self.initialize()
        self._interrupt_requested = True
        active_turn = self._active_turn_task
        if active_turn is not None and not active_turn.done():
            active_turn.cancel()
        self._terminate_active_subagents(reason="session_interrupted")
        message = self._safe_prompt(
            "interrupt",
            key="session_interrupt_message",
            fallback=_SESSION_INTERRUPT_MESSAGE,
        )
        event = CommandOutput(message=message)
        await self._persist_event(event, turn_id="")
        yield event
        completed = TurnCompleted(input_tokens=0, output_tokens=0)
        await self._persist_event(completed, turn_id="")
        yield completed

    async def send_message(self, text: str) -> AsyncIterator[Event]:
        await self.initialize()
        if self._turn_lock is None:
            raise RuntimeError("Session engine lock was not initialized")
        if self._agent is None:
            raise RuntimeError("Agent was not initialized")

        async with self._turn_lock:
            await self._apply_ready_hook_results(lock_held=True)
            turn_id = f"turn-{uuid.uuid4().hex[:10]}"
            self._active_turn_task = asyncio.current_task()
            self._active_turn_id = turn_id
            self._interrupt_requested = False
            try:
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
            finally:
                if self._active_turn_id == turn_id:
                    self._active_turn_task = None
                    self._active_turn_id = ""
                    self._interrupt_requested = False
                    self._schedule_ready_hook_apply()

    async def list_sessions(self) -> list[dict[str, Any]]:
        await self.initialize()
        if self._store is None:
            return []
        records = await self._store.list_sessions()
        return [record.model_dump() for record in records]

    async def restore_session(self, session_id: str) -> AsyncIterator[Event]:
        await self.initialize()
        await self._apply_ready_hook_results()
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
        async for event in engine_stream.stream_agent_response(
            self,
            prompt_text=prompt_text,
            turn_id=turn_id,
            persist_user=persist_user,
        ):
            yield event

    async def _run_blocking_hooks(
        self,
        *,
        turn_input_tokens: int,
        turn_output_tokens: int,
        tool_output_candidates: dict[str, tuple[str, str]],
    ) -> str | None:
        return await engine_hooks.run_blocking_hooks(
            self,
            turn_input_tokens=turn_input_tokens,
            turn_output_tokens=turn_output_tokens,
            tool_output_candidates=tool_output_candidates,
        )

    def _enqueue_background_hooks(
        self,
        *,
        turn_id: str,
        turn_input_tokens: int,
        turn_output_tokens: int,
        tool_output_candidates: dict[str, tuple[str, str]],
    ) -> None:
        engine_hooks.enqueue_background_hooks(
            self,
            turn_id=turn_id,
            turn_input_tokens=turn_input_tokens,
            turn_output_tokens=turn_output_tokens,
            tool_output_candidates=tool_output_candidates,
        )

    def _enqueue_background_hook(self, job: engine_hooks.HookJob) -> None:
        engine_hooks.enqueue_background_hook(self, job)

    async def _run_background_hook_loop(self, hook: Hook, *, hook_name: str) -> None:
        await engine_hooks.run_background_hook_loop(self, hook, hook_name=hook_name)

    def _schedule_ready_hook_apply(self) -> None:
        engine_hooks.schedule_ready_hook_apply(self)

    async def _apply_ready_hook_results(self, *, lock_held: bool = False) -> None:
        await engine_hooks.apply_ready_hook_results(self, lock_held=lock_held)

    async def _apply_ready_hook_results_unlocked(self) -> None:
        await engine_hooks.apply_ready_hook_results_unlocked(self)

    def _build_hook_context(
        self,
        *,
        turn_input_tokens: int,
        turn_output_tokens: int,
        tool_output_candidates: dict[str, tuple[str, str]],
    ) -> HookContext | None:
        return engine_hooks.build_hook_context(
            self,
            turn_input_tokens=turn_input_tokens,
            turn_output_tokens=turn_output_tokens,
            tool_output_candidates=tool_output_candidates,
        )

    def _track_background_task(self, task: asyncio.Task[Any]) -> None:
        engine_hooks.track_background_task(self, task)

    def _hook_execution_mode(self, hook: Hook) -> HookExecutionMode:
        return engine_hooks.hook_execution_mode(hook)

    def _hook_name(self, hook: Hook) -> str:
        return engine_hooks.hook_name(hook)

    def _emit_hook_update(
        self,
        *,
        hook_name: str,
        stage: HookStage,
        turn_id: str | None,
        message: str | None,
    ) -> None:
        engine_hooks.emit_hook_update(
            self,
            hook_name=hook_name,
            stage=stage,
            turn_id=turn_id,
            message=message,
        )

    def _publish_async_event(self, event: Event, *, turn_id: str) -> None:
        for queue in list(self._async_event_subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                with suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
                with suppress(asyncio.QueueFull):
                    queue.put_nowait(event)
        if self._store is None:
            return
        task = asyncio.create_task(self._persist_event(event, turn_id=turn_id))
        self._track_background_task(task)

    def _build_subagent_status_event(
        self,
        *,
        previous_snapshot: str,
    ) -> tuple[SubagentStatusUpdate | None, str | None]:
        return engine_stream.build_subagent_status_event(
            self,
            previous_snapshot=previous_snapshot,
        )

    def _build_usage_event(
        self,
        *,
        turn_input_tokens: int,
        turn_output_tokens: int,
        include_pending_turn: bool,
    ) -> UsageUpdate:
        return engine_stream.build_usage_event(
            self,
            turn_input_tokens=turn_input_tokens,
            turn_output_tokens=turn_output_tokens,
            include_pending_turn=include_pending_turn,
        )

    def _turn_cost(self, input_tokens: int, output_tokens: int) -> float:
        return engine_stream.turn_cost(self, input_tokens, output_tokens)

    def _subagent_usage_snapshot(self) -> tuple[int, int, float]:
        return engine_stream.subagent_usage_snapshot(self)

    def _terminate_active_subagents(self, *, reason: str) -> None:
        engine_stream.terminate_active_subagents(self, reason=reason)

    def _safe_prompt(self, name: str, *, key: str, fallback: str) -> str:
        return engine_stream.safe_prompt(self, name, key=key, fallback=fallback)

    async def _persist_message(
        self,
        *,
        role: str,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
        usage_input_tokens: int = 0,
        usage_output_tokens: int = 0,
    ) -> None:
        await engine_persistence.persist_message(
            self,
            role=role,
            content=content,
            tool_calls=tool_calls,
            usage_input_tokens=usage_input_tokens,
            usage_output_tokens=usage_output_tokens,
        )

    async def _persist_usage(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        await engine_persistence.persist_usage(
            self,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    async def _persist_event(self, event: Event, *, turn_id: str) -> None:
        await engine_persistence.persist_event(self, event, turn_id=turn_id)

    async def _restore_messages_to_agent(self, records: list[Any]) -> None:
        await engine_persistence.restore_messages_to_agent(self, records)
