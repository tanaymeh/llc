from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from llc.agent.hooks import Hook, HookContext, HookExecutionMode, hook_order
from llc.service.events import HookStage, HookUpdate


@dataclass(slots=True)
class HookJob:
    hook: Hook
    hook_name: str
    turn_id: str | None
    ctx: HookContext


@dataclass(slots=True)
class HookPreparedResult:
    hook: Hook
    hook_name: str
    turn_id: str | None
    ctx: HookContext
    prepared: Any


async def run_blocking_hooks(
    engine: Any,
    *,
    turn_input_tokens: int,
    turn_output_tokens: int,
    tool_output_candidates: dict[str, tuple[str, str]],
) -> str | None:
    if engine._agent is None:
        return None
    if (
        turn_input_tokens <= 0
        and turn_output_tokens <= 0
        and not tool_output_candidates
    ):
        return None

    hook_ctx = build_hook_context(
        engine,
        turn_input_tokens=turn_input_tokens,
        turn_output_tokens=turn_output_tokens,
        tool_output_candidates=tool_output_candidates,
    )
    if hook_ctx is None:
        return None
    messages: list[str] = []
    for hook in engine._blocking_hooks:
        try:
            prepared = await hook.prepare_after_turn(hook_ctx)
        except Exception as exc:  # noqa: BLE001
            messages.append(f"Hook failed: `{hook_name(hook)}`: `{exc!s}`")
            continue
        if prepared is None:
            continue
        try:
            maybe_message = await hook.apply_prepared(hook_ctx, prepared)
        except Exception as exc:  # noqa: BLE001
            messages.append(f"Hook failed: `{hook_name(hook)}`: `{exc!s}`")
            continue
        if maybe_message:
            messages.append(maybe_message)
    return "\n".join(messages) if messages else None


def enqueue_background_hooks(
    engine: Any,
    *,
    turn_id: str,
    turn_input_tokens: int,
    turn_output_tokens: int,
    tool_output_candidates: dict[str, tuple[str, str]],
) -> None:
    if engine._agent is None or not engine._background_hooks:
        return
    if (
        turn_input_tokens <= 0
        and turn_output_tokens <= 0
        and not tool_output_candidates
    ):
        return
    hook_ctx = build_hook_context(
        engine,
        turn_input_tokens=turn_input_tokens,
        turn_output_tokens=turn_output_tokens,
        tool_output_candidates=tool_output_candidates,
    )
    if hook_ctx is None:
        return
    for hook in engine._background_hooks:
        enqueue_background_hook(
            engine,
            HookJob(
                hook=hook,
                hook_name=hook_name(hook),
                turn_id=turn_id,
                ctx=hook_ctx,
            ),
        )


def enqueue_background_hook(engine: Any, job: HookJob) -> None:
    hook_key = job.hook_name
    pending_existing = engine._hook_pending_jobs.get(hook_key)
    engine._hook_pending_jobs[hook_key] = job
    if pending_existing is not None:
        emit_hook_update(
            engine,
            hook_name=hook_key,
            stage="coalesced",
            turn_id=job.turn_id,
            message="Replaced older queued hook run.",
        )
    emit_hook_update(
        engine,
        hook_name=hook_key,
        stage="queued",
        turn_id=job.turn_id,
        message=None,
    )

    runner = engine._hook_runner_tasks.get(hook_key)
    if runner is not None and not runner.done():
        return
    task = asyncio.create_task(run_background_hook_loop(engine, job.hook, hook_name=hook_key))
    engine._hook_runner_tasks[hook_key] = task
    track_background_task(engine, task)


async def run_background_hook_loop(engine: Any, hook: Hook, *, hook_name: str) -> None:
    while True:
        job = engine._hook_pending_jobs.pop(hook_name, None)
        if job is None:
            return

        emit_hook_update(
            engine,
            hook_name=hook_name,
            stage="running",
            turn_id=job.turn_id,
            message=None,
        )
        try:
            prepared = await hook.prepare_after_turn(job.ctx)
        except Exception as exc:  # noqa: BLE001
            emit_hook_update(
                engine,
                hook_name=hook_name,
                stage="failed",
                turn_id=job.turn_id,
                message=f"{exc!s}",
            )
            continue

        if prepared is None:
            emit_hook_update(
                engine,
                hook_name=hook_name,
                stage="skipped",
                turn_id=job.turn_id,
                message=None,
            )
            continue

        if hook_name in engine._hook_ready_results:
            emit_hook_update(
                engine,
                hook_name=hook_name,
                stage="coalesced",
                turn_id=job.turn_id,
                message="Replaced older prepared hook result.",
            )
        engine._hook_ready_results[hook_name] = HookPreparedResult(
            hook=hook,
            hook_name=hook_name,
            turn_id=job.turn_id,
            ctx=job.ctx,
            prepared=prepared,
        )
        schedule_ready_hook_apply(engine)


def schedule_ready_hook_apply(engine: Any) -> None:
    if not engine._hook_ready_results:
        return
    task = engine._hook_apply_task
    if task is not None and not task.done():
        return
    task = asyncio.create_task(apply_ready_hook_results(engine))
    engine._hook_apply_task = task
    track_background_task(engine, task)


async def apply_ready_hook_results(engine: Any, *, lock_held: bool = False) -> None:
    if not engine._hook_ready_results:
        return
    if engine._agent is None:
        engine._hook_ready_results.clear()
        return

    if lock_held:
        await apply_ready_hook_results_unlocked(engine)
        return

    if engine._active_turn_task is not None and not engine._active_turn_task.done():
        return
    if engine._turn_lock is None:
        return
    async with engine._turn_lock:
        await apply_ready_hook_results_unlocked(engine)


async def apply_ready_hook_results_unlocked(engine: Any) -> None:
    if not engine._hook_ready_results:
        return
    async with engine._hook_apply_lock:
        if engine._active_turn_task is not None and not engine._active_turn_task.done():
            return
        for hook in sorted(engine._background_hooks, key=hook_order):
            hook_key = hook_name(hook)
            ready = engine._hook_ready_results.pop(hook_key, None)
            if ready is None:
                continue
            try:
                maybe_message = await ready.hook.apply_prepared(ready.ctx, ready.prepared)
            except Exception as exc:  # noqa: BLE001
                emit_hook_update(
                    engine,
                    hook_name=hook_key,
                    stage="failed",
                    turn_id=ready.turn_id,
                    message=f"{exc!s}",
                )
                continue
            if maybe_message:
                emit_hook_update(
                    engine,
                    hook_name=hook_key,
                    stage="applied",
                    turn_id=ready.turn_id,
                    message=maybe_message,
                )
            else:
                emit_hook_update(
                    engine,
                    hook_name=hook_key,
                    stage="skipped",
                    turn_id=ready.turn_id,
                    message=None,
                )
    if engine._hook_ready_results:
        schedule_ready_hook_apply(engine)


def build_hook_context(
    engine: Any,
    *,
    turn_input_tokens: int,
    turn_output_tokens: int,
    tool_output_candidates: dict[str, tuple[str, str]],
) -> HookContext | None:
    if engine._agent is None:
        return None
    return HookContext(
        agent=engine._agent,
        thread_id=engine._thread_id,
        settings=engine._settings,
        last_turn_input_tokens=turn_input_tokens,
        last_turn_output_tokens=turn_output_tokens,
        available_models=engine._available_models,
        compact_prompt=engine._settings.compact_prompt,
        tool_output_candidates=tool_output_candidates,
    )


def track_background_task(engine: Any, task: asyncio.Task[Any]) -> None:
    engine._background_tasks.add(task)

    def _cleanup(done: asyncio.Task[Any]) -> None:
        engine._background_tasks.discard(done)
        if engine._hook_apply_task is done:
            engine._hook_apply_task = None
        stale_runner_names = [
            name
            for name, runner in engine._hook_runner_tasks.items()
            if runner is done
        ]
        for name in stale_runner_names:
            engine._hook_runner_tasks.pop(name, None)
        try:
            done.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    task.add_done_callback(_cleanup)


def hook_execution_mode(hook: Hook) -> HookExecutionMode:
    mode = str(getattr(hook, "execution_mode", "background")).strip().lower()
    if mode == "blocking":
        return "blocking"
    return "background"


def hook_name(hook: Hook) -> str:
    raw_name = str(getattr(hook, "name", hook.__class__.__name__)).strip()
    if raw_name:
        return raw_name
    return hook.__class__.__name__


def emit_hook_update(
    engine: Any,
    *,
    hook_name: str,
    stage: HookStage,
    turn_id: str | None,
    message: str | None,
) -> None:
    event = HookUpdate(
        hook_name=hook_name,
        stage=stage,
        turn_id=turn_id,
        message=message,
    )
    engine._publish_async_event(event, turn_id=turn_id or "")
