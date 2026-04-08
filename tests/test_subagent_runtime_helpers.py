from __future__ import annotations

import unittest

from llc.agent.subagents import SubAgentRuntime
from llc.agent.subagents.reporting import (
    record_snapshot,
    stop_reason_is_stuck,
    stuck_activity_detail,
    stuck_latest_report,
)
from llc.agent.subagents.types import SubAgentRecord
from llc.agent.subagents.worker_runner import task_with_feedback
from llc.config import Settings


class SubagentRuntimeHelperTests(unittest.TestCase):
    def test_launch_requires_explicit_name(self) -> None:
        runtime = SubAgentRuntime(Settings(), build_subagent=lambda settings: None)
        result = runtime.launch_subagent("Investigate issue", source="orchestrator")
        self.assertFalse(result["ok"])
        self.assertIn("name", str(result.get("error", "")).lower())

    def test_launch_uses_provided_name(self) -> None:
        runtime = SubAgentRuntime(Settings(), build_subagent=lambda settings: None)
        original_submit = runtime._submit_locked
        runtime._submit_locked = lambda record: None  # type: ignore[method-assign]
        try:
            result = runtime.launch_subagent(
                "Investigate issue",
                source="orchestrator",
                name="planner",
            )
        finally:
            runtime._submit_locked = original_submit  # type: ignore[method-assign]
        self.assertTrue(result["ok"])
        self.assertEqual(result["name"], "planner")

    def test_stop_reason_stuck_detection_handles_suffixes(self) -> None:
        self.assertTrue(stop_reason_is_stuck("max_runtime_exceeded"))
        self.assertTrue(stop_reason_is_stuck("stall_timeout_exceeded:unresponsive_after_stop"))
        self.assertFalse(stop_reason_is_stuck("terminated:user_request"))

    def test_stuck_human_messages_cover_known_reasons(self) -> None:
        self.assertEqual(
            stuck_latest_report("max_tool_calls_exceeded"),
            "Tool-call budget exceeded. Stopping.",
        )
        self.assertEqual(
            stuck_activity_detail("stall_timeout_exceeded"),
            "Stopped after no progress heartbeat.",
        )

    def test_record_snapshot_completed_includes_final_result(self) -> None:
        record = SubAgentRecord(
            id="subagent-test-1",
            name="Mr Darcy",
            base_task="Analyze project",
            task="Analyze project",
            context="",
            source="orchestrator",
            thread_id="thread-1",
            created_at=1.0,
            updated_at=2.0,
            status="completed",
            final_output="Done successfully.",
        )
        snapshot = record_snapshot(
            record,
            task_preview_chars=180,
            activity_preview_chars=80,
            report_preview_chars=220,
            error_preview_chars=220,
            final_output_preview_chars=600,
        )
        self.assertEqual(snapshot["status"], "completed")
        self.assertEqual(snapshot["final_result"], "Done successfully.")
        self.assertEqual(snapshot["goal"], "Analyze project")

    def test_task_with_feedback_merges_prior_and_feedback(self) -> None:
        merged = task_with_feedback(
            "Write a summary",
            "Make it shorter",
            "A very long first attempt.",
            prompt_registry=None,
        )
        self.assertIn("Original task", merged)
        self.assertIn("Prior attempt output", merged)
        self.assertIn("Revision instructions", merged)
        self.assertIn("Produce an improved final result.", merged)

    def test_history_persistence_skips_messages_already_captured_live(self) -> None:
        emitted_messages: list[dict[str, object]] = []
        runtime = SubAgentRuntime(
            Settings(),
            build_subagent=lambda settings: None,
            conversation_id="conv-123",
            message_sink=lambda **payload: emitted_messages.append(payload),
        )
        record = SubAgentRecord(
            id="subagent-test-2",
            conversation_id="conv-123",
            name="planner",
            base_task="Analyze project",
            task="Analyze project",
            context="",
            source="orchestrator",
            thread_id="thread-2",
            parent_turn_id="turn-1",
            created_at=1.0,
            updated_at=2.0,
        )
        runtime._persist_worker_history_locked(  # type: ignore[attr-defined]
            record,
            [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "task prompt"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "tool-1", "name": "Read", "args": {}}],
                },
                {
                    "role": "tool",
                    "content": "tool result",
                    "tool_call_id": "tool-1",
                },
                {
                    "role": "assistant",
                    "content": "Need one more check before finishing.",
                    "tool_calls": [{"id": "tool-2", "name": "LS", "args": {}}],
                },
                {"role": "assistant", "content": "Final answer"},
            ],
            final_output="Final answer",
        )
        self.assertEqual(len(emitted_messages), 1)
        self.assertEqual(
            emitted_messages[0]["content"],
            "Need one more check before finishing.",
        )
        self.assertEqual(emitted_messages[0]["message_kind"], "history_snapshot")


if __name__ == "__main__":
    unittest.main()
