from __future__ import annotations

import asyncio
import unittest
from typing import Any

from llc.commands import Command, CommandRegistry, CommandResult, ReplContext
from llc.config import Settings
from llc.service.engine import SessionEngine
from llc.service.events import HookUpdate, TurnCompleted


class _PingCommand(Command):
    @property
    def name(self) -> str:
        return "/ping"

    @property
    def description(self) -> str:
        return "No-op command used for tests."

    async def execute(self, args: str, ctx: ReplContext) -> CommandResult:
        del args, ctx
        return CommandResult(message="pong")


class _ControlledBackgroundHook:
    name = "controlled_hook"
    order = 10
    execution_mode = "background"

    def __init__(self, *, block_first_prepare: bool = True) -> None:
        self._block_first_prepare = block_first_prepare
        self.first_prepare_started = asyncio.Event()
        self.release_first_prepare = asyncio.Event()
        self.prepare_inputs: list[int] = []
        self.applied_inputs: list[int] = []

    async def prepare_after_turn(self, ctx: Any) -> dict[str, int]:
        self.prepare_inputs.append(int(ctx.last_turn_input_tokens))
        if self._block_first_prepare and len(self.prepare_inputs) == 1:
            self.first_prepare_started.set()
            await self.release_first_prepare.wait()
        return {"input_tokens": int(ctx.last_turn_input_tokens)}

    async def apply_prepared(self, ctx: Any, prepared: dict[str, int]) -> str | None:
        del ctx
        value = int(prepared.get("input_tokens", 0))
        self.applied_inputs.append(value)
        return f"Applied {value}"


class HookRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def _build_engine(self, *, registry: CommandRegistry | None = None) -> SessionEngine:
        engine = SessionEngine(Settings(), registry or CommandRegistry())
        engine._initialized = True
        engine._turn_lock = asyncio.Lock()
        engine._agent = object()
        return engine

    async def _drain_until_applied(
        self,
        queue: asyncio.Queue[Any],
        *,
        timeout_s: float = 3.0,
    ) -> list[HookUpdate]:
        deadline = asyncio.get_running_loop().time() + timeout_s
        updates: list[HookUpdate] = []
        while asyncio.get_running_loop().time() < deadline:
            remaining = max(deadline - asyncio.get_running_loop().time(), 0.05)
            event = await asyncio.wait_for(queue.get(), timeout=remaining)
            if isinstance(event, HookUpdate):
                updates.append(event)
                if event.stage == "applied":
                    return updates
        self.fail("Timed out waiting for hook applied event")

    async def test_background_hook_emits_lifecycle_and_applies(self) -> None:
        hook = _ControlledBackgroundHook()
        engine = self._build_engine()
        engine._hooks = [hook]
        engine._blocking_hooks = []
        engine._background_hooks = [hook]

        queue = engine.subscribe_async_events()
        try:
            engine._enqueue_background_hooks(
                turn_id="turn-1",
                turn_input_tokens=11,
                turn_output_tokens=4,
                tool_output_candidates={},
            )
            await asyncio.wait_for(hook.first_prepare_started.wait(), timeout=1.0)
            hook.release_first_prepare.set()
            updates = await self._drain_until_applied(queue)
            self.assertIn("queued", [event.stage for event in updates])
            self.assertIn("running", [event.stage for event in updates])
            self.assertIn("applied", [event.stage for event in updates])
            self.assertEqual(hook.applied_inputs, [11])
        finally:
            engine.unsubscribe_async_events(queue)
            await engine.shutdown()

    async def test_background_hook_coalesces_latest_pending_job(self) -> None:
        hook = _ControlledBackgroundHook()
        engine = self._build_engine()
        engine._hooks = [hook]
        engine._blocking_hooks = []
        engine._background_hooks = [hook]

        queue = engine.subscribe_async_events()
        try:
            engine._enqueue_background_hooks(
                turn_id="turn-1",
                turn_input_tokens=10,
                turn_output_tokens=1,
                tool_output_candidates={},
            )
            await asyncio.wait_for(hook.first_prepare_started.wait(), timeout=1.0)

            engine._enqueue_background_hooks(
                turn_id="turn-2",
                turn_input_tokens=20,
                turn_output_tokens=1,
                tool_output_candidates={},
            )
            engine._enqueue_background_hooks(
                turn_id="turn-3",
                turn_input_tokens=30,
                turn_output_tokens=1,
                tool_output_candidates={},
            )

            hook.release_first_prepare.set()
            deadline = asyncio.get_running_loop().time() + 3.0
            while asyncio.get_running_loop().time() < deadline:
                if hook.applied_inputs and hook.applied_inputs[-1] == 30:
                    break
                await asyncio.sleep(0.02)
            self.assertEqual(hook.prepare_inputs, [10, 30])
            self.assertEqual(hook.applied_inputs[-1:], [30])

            updates: list[HookUpdate] = []
            while not queue.empty():
                event = queue.get_nowait()
                if isinstance(event, HookUpdate):
                    updates.append(event)
            self.assertTrue(any(event.stage == "coalesced" for event in updates))
        finally:
            engine.unsubscribe_async_events(queue)
            await engine.shutdown()

    async def test_ready_hook_result_waits_for_safe_point(self) -> None:
        hook = _ControlledBackgroundHook(block_first_prepare=False)
        engine = self._build_engine()
        engine._hooks = [hook]
        engine._blocking_hooks = []
        engine._background_hooks = [hook]

        hold_turn = asyncio.Event()
        active_turn_task = asyncio.create_task(hold_turn.wait())
        engine._active_turn_task = active_turn_task
        engine._active_turn_id = "turn-running"

        try:
            engine._enqueue_background_hooks(
                turn_id="turn-1",
                turn_input_tokens=15,
                turn_output_tokens=5,
                tool_output_candidates={},
            )
            await asyncio.sleep(0.1)
            self.assertEqual(hook.applied_inputs, [])
            self.assertIn(hook.name, engine._hook_ready_results)

            hold_turn.set()
            await active_turn_task
            engine._active_turn_task = None
            engine._active_turn_id = ""
            await engine._apply_ready_hook_results()
            self.assertEqual(hook.applied_inputs, [15])
        finally:
            await engine.shutdown()

    async def test_send_message_is_not_blocked_by_running_background_hook(self) -> None:
        registry = CommandRegistry()
        registry.register(_PingCommand())
        hook = _ControlledBackgroundHook()
        engine = self._build_engine(registry=registry)
        engine._hooks = [hook]
        engine._blocking_hooks = []
        engine._background_hooks = [hook]

        try:
            engine._enqueue_background_hooks(
                turn_id="turn-prev",
                turn_input_tokens=99,
                turn_output_tokens=1,
                tool_output_candidates={},
            )
            await asyncio.wait_for(hook.first_prepare_started.wait(), timeout=1.0)

            started_at = asyncio.get_running_loop().time()
            events = [event async for event in engine.send_message("/ping")]
            elapsed_s = asyncio.get_running_loop().time() - started_at

            self.assertLess(elapsed_s, 1.0)
            self.assertTrue(any(isinstance(event, TurnCompleted) for event in events))
        finally:
            hook.release_first_prepare.set()
            await engine.shutdown()


if __name__ == "__main__":
    unittest.main()
