from __future__ import annotations

import asyncio
import json
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

_SESSION_TITLE_MAX_LEN = 80


def _derive_session_title(text: str) -> str:
    first_line = text.strip().split("\n", 1)[0].strip()
    if len(first_line) <= _SESSION_TITLE_MAX_LEN:
        return first_line
    truncated = first_line[:_SESSION_TITLE_MAX_LEN].rsplit(" ", 1)[0]
    return f"{truncated}..." if truncated else f"{first_line[:_SESSION_TITLE_MAX_LEN]}..."


class SessionEngine:
    def __init__(
        self,
        settings: Settings,
        registry: CommandRegistry,
        *,
        conversation_id: str | None = None,
        session_id: str | None = None,
        store: Any | None = None,
        enable_langfuse_tracing: bool = False,
    ) -> None:
        self._settings = settings
        self._registry = registry
        self._store = store
        requested_conversation_id = (conversation_id or session_id or "").strip()
        self._conversation_id = requested_conversation_id or str(uuid.uuid4())
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
        self._db_persisted = False
        self._persist_queue: asyncio.Queue[
            engine_persistence.PersistenceJob
        ] | None = None
        self._persist_writer_task: asyncio.Task[Any] | None = None

    @property
    def conversation_id(self) -> str:
        return self._conversation_id

    @property
    def session_id(self) -> str:
        return self._conversation_id

    @property
    def model_name(self) -> str:
        return self._settings.model_name

    @property
    def sub_agent_mode_enabled(self) -> bool:
        return self._settings.sub_agent_mode_enabled

    @property
    def available_models(self) -> list[AvailableModel]:
        return list(self._available_models)

    def _subagent_launch_precheck(self) -> str | None:
        selected_model = str(self._settings.model_name or "").strip()
        if not selected_model:
            return "No model is configured for sub-agent launches."

        models = self._available_models
        if not models:
            return None

        available_ids = {model.id for model in models if model.id}
        if selected_model in available_ids:
            return None

        suggestions = ", ".join(f"`{model.id}`" for model in models[:5])
        if suggestions:
            return (
                f"Configured model `{selected_model}` is unavailable on the current provider. "
                f"Choose an available model via `/model` (examples: {suggestions})."
            )
        return (
            f"Configured model `{selected_model}` is unavailable on the current provider. "
            "Choose an available model via `/model`."
        )

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

    def _create_subagent_runtime(self) -> SubAgentRuntime:
        return SubAgentRuntime(
            self._settings,
            build_subagent=lambda settings: build_agent_graph(
                settings,
                role="subagent",
            ),
            prompt_registry=self._prompt_registry,
            launch_precheck=self._subagent_launch_precheck,
            conversation_id=self._conversation_id,
            event_sink=self._schedule_subagent_event_persist,
            message_sink=self._schedule_subagent_message_persist,
            usage_sink=self._schedule_subagent_usage_persist,
        )

    def _create_agent_graph(self) -> Any:
        role = "orchestrator" if self._settings.sub_agent_mode_enabled else "default"
        return build_agent_graph(
            self._settings,
            role=role,
            subagent_runtime=self._subagent_runtime,
        )

    def _reset_runtime_and_agent(self) -> None:
        if self._subagent_runtime is not None:
            self._subagent_runtime.shutdown()
        self._subagent_runtime = self._create_subagent_runtime()
        self._bind_subagent_runtime()
        self._agent = self._create_agent_graph()

    def _reset_usage_accumulators(self) -> None:
        self._token_hook.session_input = 0
        self._token_hook.session_output = 0
        self._token_hook.session_cost = 0.0
        if self._subagent_runtime is not None:
            self._subagent_runtime.load_usage_totals({})

    async def _restore_usage_accumulators(self) -> None:
        self._reset_usage_accumulators()
        if self._store is None:
            return
        aggregates = await self._store.aggregate_conversation_usage(self._conversation_id)
        orchestrator_usage = aggregates.get("orchestrator", {})
        self._token_hook.session_input = max(
            int(orchestrator_usage.get("input_tokens", 0) or 0),
            0,
        )
        self._token_hook.session_output = max(
            int(orchestrator_usage.get("output_tokens", 0) or 0),
            0,
        )
        self._token_hook.session_cost = max(
            float(orchestrator_usage.get("cost", 0.0) or 0.0),
            0.0,
        )
        if self._subagent_runtime is not None:
            subagent_usage = aggregates.get("subagent", {})
            by_model = subagent_usage.get("by_model", {})
            self._subagent_runtime.load_usage_totals(
                by_model if isinstance(by_model, dict) else {}
            )

    async def _cancel_hook_tasks(self) -> None:
        tasks: list[asyncio.Task[Any]] = []
        if self._hook_apply_task is not None and not self._hook_apply_task.done():
            tasks.append(self._hook_apply_task)
        tasks.extend(
            task
            for task in self._hook_runner_tasks.values()
            if not task.done()
        )
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task
        self._hook_runner_tasks.clear()
        self._hook_pending_jobs.clear()
        self._hook_ready_results.clear()
        self._hook_apply_task = None

    async def initialize(self) -> None:
        if self._initialized:
            return

        self._turn_lock = asyncio.Lock()
        self._reset_runtime_and_agent()

        self._available_models = await fetch_models(
            base_url=self._settings.openai_base_url,
            api_key=self._settings.openai_api_key,
        )

        if self._store is not None:
            await self._store.initialize()
            await self._start_persistence_writer()
            record = await self._store.get_canonical_conversation(
                self._conversation_id,
            )
            if record is not None:
                self._settings = self._settings.model_copy(
                    update={
                        "model_name": record.model_name,
                        "sub_agent_mode_enabled": bool(record.sub_agent_mode),
                    },
                )
                self._db_persisted = True
                self._reset_runtime_and_agent()
        self._initialized = True

    async def _ensure_db_persisted(self, title: str = "") -> None:
        if self._db_persisted or self._store is None:
            return
        now = time.time()
        await self._store.ensure_conversation(
            self._conversation_id,
            model_name=self._settings.model_name,
            sub_agent_mode=self._settings.sub_agent_mode_enabled,
            title=title,
            created_at=now,
            updated_at=now,
        )
        self._db_persisted = True

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
            await self._flush_persistence_queue()
            await self._stop_persistence_writer()
            await self._store.close()
        self._initialized = False

    async def interrupt(self) -> AsyncIterator[Event]:
        await self.initialize()
        await self._ensure_db_persisted()
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
            if not self._db_persisted:
                title = _derive_session_title(text)
                await self._ensure_db_persisted(title=title)
            turn_id = f"turn-{uuid.uuid4().hex[:10]}"
            self._active_turn_task = asyncio.current_task()
            self._active_turn_id = turn_id
            self._interrupt_requested = False
            try:
                started = TurnStarted(turn_id=turn_id)
                await self._persist_event(started, turn_id=turn_id)
                yield started
                await self._persist_message(
                    role="user",
                    content=text,
                    turn_id=turn_id,
                    visible_to_orchestrator=True,
                )

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
        if self._turn_lock is None:
            raise RuntimeError("Session engine lock was not initialized")

        async with self._turn_lock:
            await self._apply_ready_hook_results(lock_held=True)
            await self._ensure_db_persisted()
            requested_conversation_id = session_id.strip()
            if requested_conversation_id != self._conversation_id:
                event = ErrorOccurred(
                    message=(
                        "Restore target does not match this engine conversation. "
                        "Switch to the target conversation engine first."
                    )
                )
                await self._persist_event(event, turn_id="")
                yield event
                return
            if self._store is None:
                event = ErrorOccurred(message="Session store is not configured.")
                await self._persist_event(event, turn_id="")
                yield event
                return
            runtime = self._subagent_runtime
            if runtime is not None:
                try:
                    active_subagents = int(
                        runtime.get_subagent_report().get("active_count", 0) or 0
                    )
                except Exception:
                    active_subagents = 0
                if active_subagents > 0:
                    event = ErrorOccurred(
                        message="Cannot restore while sub-agents are still active."
                    )
                    await self._persist_event(event, turn_id="")
                    yield event
                    return

            snapshot = await self._store.get_state_snapshot(self._conversation_id)
            visible_messages = await self._store.list_conversation_messages(
                self._conversation_id,
                visible_to_orchestrator=True,
            )
            legacy_messages: list[Any] | None = None
            if snapshot is not None:
                try:
                    parsed_snapshot = json.loads(snapshot.payload_json)
                except Exception:
                    parsed_snapshot = []
                count = len(parsed_snapshot) if isinstance(parsed_snapshot, list) else 0
            else:
                count = len(visible_messages)
            if count == 0:
                legacy_messages = await self._store.get_legacy_messages(
                    self._conversation_id
                )
                count = len(legacy_messages)
            if count == 0:
                event = ErrorOccurred(
                    message=f"No session data found for `{self._conversation_id}`."
                )
                await self._persist_event(event, turn_id="")
                yield event
                return

            await self._cancel_hook_tasks()
            self._reset_runtime_and_agent()
            await self._restore_messages_to_agent(legacy_messages)
            await self._restore_usage_accumulators()
            event = SessionRestored(
                conversation_id=self._conversation_id,
                session_id=self._conversation_id,
                message_count=count,
            )
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
        )
        result = await command.execute(args, context)
        self._settings = context.settings
        self._agent = context.agent
        self._subagent_runtime = context.subagent_runtime
        self._bind_subagent_runtime()

        if self._store is not None:
            await self._store.update_session(
                self._conversation_id,
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
            await self._persist_message(
                role="assistant",
                content=result.message,
                turn_id=turn_id,
                message_kind="command",
                visible_to_orchestrator=True,
            )

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
        await self._persist_orchestrator_snapshot()

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

    def _bind_subagent_runtime(self) -> None:
        runtime = self._subagent_runtime
        if runtime is None:
            return
        runtime.set_conversation_id(self._conversation_id)
        runtime.set_artifact_sinks(
            event_sink=self._schedule_subagent_event_persist,
            message_sink=self._schedule_subagent_message_persist,
            usage_sink=self._schedule_subagent_usage_persist,
        )
        runtime.update_settings(self._settings)

    def _schedule_store_persist(self, coro: Any) -> None:
        try:
            task = asyncio.create_task(coro)
        except RuntimeError:
            return
        self._track_background_task(task)

    def _schedule_subagent_event_persist(self, **payload: Any) -> None:
        if self._store is None:
            return
        self._schedule_store_persist(
            self._persist_event(
                turn_id=str(payload.get("turn_id", "") or ""),
                actor_kind=str(payload.get("actor_kind", "") or "subagent"),
                actor_id=str(payload.get("actor_id", "") or ""),
                event_type=str(payload.get("event_type", "") or ""),
                payload=payload.get("payload")
                if isinstance(payload.get("payload"), dict)
                else {},
                langfuse_trace_id=str(payload.get("langfuse_trace_id", "") or ""),
            )
        )

    def _schedule_subagent_message_persist(self, **payload: Any) -> None:
        if self._store is None:
            return
        raw_tool_calls = payload.get("tool_calls")
        tool_calls = raw_tool_calls if isinstance(raw_tool_calls, list) else None
        self._schedule_store_persist(
            self._persist_message(
                role=str(payload.get("role", "") or "assistant"),
                content=str(payload.get("content", "") or ""),
                tool_calls=tool_calls,
                usage_input_tokens=int(payload.get("usage_input_tokens", 0) or 0),
                usage_output_tokens=int(payload.get("usage_output_tokens", 0) or 0),
                actor_kind=str(payload.get("actor_kind", "") or "subagent"),
                actor_id=str(payload.get("actor_id", "") or ""),
                turn_id=str(payload.get("turn_id", "") or ""),
                message_kind=str(payload.get("message_kind", "") or "chat"),
                tool_call_id=str(payload.get("tool_call_id", "") or ""),
                visible_to_orchestrator=bool(
                    payload.get("visible_to_orchestrator", False)
                ),
                langfuse_trace_id=str(payload.get("langfuse_trace_id", "") or ""),
            )
        )

    def _schedule_subagent_usage_persist(self, **payload: Any) -> None:
        if self._store is None:
            return
        self._schedule_store_persist(
            self._persist_usage(
                input_tokens=int(payload.get("input_tokens", 0) or 0),
                output_tokens=int(payload.get("output_tokens", 0) or 0),
                actor_kind=str(payload.get("actor_kind", "") or "subagent"),
                actor_id=str(payload.get("actor_id", "") or ""),
            )
        )

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

    def _new_persist_ack(self) -> asyncio.Future[None]:
        return asyncio.get_running_loop().create_future()

    async def _start_persistence_writer(self) -> None:
        if self._store is None or self._persist_writer_task is not None:
            return
        self._persist_queue = asyncio.Queue(maxsize=self._settings.persistence_queue_maxsize)
        self._persist_writer_task = asyncio.create_task(
            self._run_persistence_writer()
        )

    async def _run_persistence_writer(self) -> None:
        queue = self._persist_queue
        if queue is None:
            return
        
        while True:
            job = await queue.get()
            try:
                if job.kind in {"flush", "stop"}:
                    if job.ack is not None and not job.ack.done():
                        job.ack.set_result(None)
                    if job.kind == "stop":
                        return
                    continue
                if job.kind == "message":
                    await engine_persistence.persist_message(self, **job.kwargs)
                elif job.kind == "event":
                    await engine_persistence.persist_event(
                        self,
                        job.event,
                        **job.kwargs
                    )
                elif job.kind == "usage":
                    await engine_persistence.persist_usage(self, **job.kwargs)
                else:
                    raise RuntimeError(f"Unknown persistence job kind: {job.kind}")
                

                if job.ack is not None and not job.ack.done():
                    job.ack.set_result(None)
            
            except Exception as exc:
                if job.ack is not None and not job.ack.done():
                    job.ack.set_exception(exc)

            finally:
                queue.task_done()

    async def _enqueue_persistence_job(
        self,
        job: engine_persistence.PersistenceJob
    ) -> None:
        if self._store is None:
            return
        if self._persist_queue is None:
            await self._start_persistence_writer()
        queue = self._persist_queue
        if queue is None:
            raise RuntimeError("Persistence queue is unavailable")
        await queue.put(job)
        if job.ack is not None:
            await job.ack

    async def _flush_persistence_queue(self) -> None:
        if self._store is None or self._persist_writer_task is None:
            return
        await self._enqueue_persistence_job(
            engine_persistence.build_persistence_job(
                "flush",
                ack=self._new_persist_ack(),
            )
        )

    async def _stop_persistence_writer(self) -> None:
        task = self._persist_writer_task
        if task is None:
            return
        await self._enqueue_persistence_job(
            engine_persistence.build_persistence_job(
                "stop",
                ack=self._new_persist_ack()
            )
        )

        await task
        self._persist_writer_task = None
        self._persist_queue = None

    async def _persist_message(
        self,
        *,
        role: str,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
        usage_input_tokens: int = 0,
        usage_output_tokens: int = 0,
        actor_kind: str = "orchestrator",
        actor_id: str = "orchestrator",
        turn_id: str = "",
        message_kind: str = "chat",
        tool_call_id: str = "",
        visible_to_orchestrator: bool = False,
        langfuse_trace_id: str = "",
        wait_for_ack: bool = True,
    ) -> None:
        ack = self._new_persist_ack() if wait_for_ack else None
        await self._enqueue_persistence_job(
            engine_persistence.build_persistence_job(
                "message",
                ack=ack,
                role=role,
                content=content,
                tool_calls=tool_calls,
                usage_input_tokens=usage_input_tokens,
                usage_output_tokens=usage_output_tokens,
                actor_kind=actor_kind,
                actor_id=actor_id,
                turn_id=turn_id,
                message_kind=message_kind,
                tool_call_id=tool_call_id,
                visible_to_orchestrator=visible_to_orchestrator,
                langfuse_trace_id=langfuse_trace_id,
            )
        )

    async def _persist_usage(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        actor_kind: str = "orchestrator",
        actor_id: str = "orchestrator",
        wait_for_ack: bool = True,
    ) -> None:
        ack = self._new_persist_ack() if wait_for_ack else None
        await self._enqueue_persistence_job(
            engine_persistence.build_persistence_job(
                "usage",
                ack=ack,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                actor_kind=actor_kind,
                actor_id=actor_id,
            )
        )

    async def _persist_event(
        self,
        event: Event | None = None,
        *,
        turn_id: str,
        actor_kind: str | None = None,
        actor_id: str | None = None,
        event_type: str | None = None,
        payload: dict[str, Any] | None = None,
        langfuse_trace_id: str = "",
        wait_for_ack: bool = True,
    ) -> None:
        ack = self._new_persist_ack() if wait_for_ack else None
        await self._enqueue_persistence_job(
            engine_persistence.build_persistence_job(
                "event",
                event=event,
                ack=ack,
                turn_id=turn_id,
                actor_kind=actor_kind,
                actor_id=actor_id,
                event_type=event_type,
                payload=payload,
                langfuse_trace_id=langfuse_trace_id,
            )
        )

    async def _persist_orchestrator_snapshot(self) -> None:
        await engine_persistence.persist_orchestrator_snapshot(self)

    async def _restore_messages_to_agent(self, records: list[Any] | None = None) -> None:
        await engine_persistence.restore_messages_to_agent(self, records)
