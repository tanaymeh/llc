from __future__ import annotations

import multiprocessing as mp
import unittest

from llc.agent.subagents.coordination import (
    SubAgentCoordinationClient,
    SubAgentCoordinationHub,
)
from llc.agent.tools import collect_tools
from llc.config import Settings


class SubAgentCoordinationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx = mp.get_context("spawn")
        self.hub = SubAgentCoordinationHub(self.ctx)
        self.hub.register_worker("worker-a", name="A", goal="Analyze API")
        self.hub.register_worker("worker-b", name="B", goal="Analyze DB")
        payload_a = self.hub.worker_payload(
            "worker-a",
            default_wait_timeout_ms=300,
            default_claim_ttl_s=120,
            stall_timeout_s=45,
        )
        payload_b = self.hub.worker_payload(
            "worker-b",
            default_wait_timeout_ms=300,
            default_claim_ttl_s=120,
            stall_timeout_s=45,
        )
        self.client_a = SubAgentCoordinationClient.from_payload(payload_a)
        self.client_b = SubAgentCoordinationClient.from_payload(payload_b)
        if self.client_a is None or self.client_b is None:
            raise RuntimeError("Failed to build coordination clients in tests.")

    def tearDown(self) -> None:
        self.hub.shutdown()

    def test_submit_plan_requires_multi_step_payload(self) -> None:
        fail_payload = self.client_a.submit_plan(["one"])
        self.assertFalse(fail_payload["ok"])
        ok_payload = self.client_a.submit_plan(["inspect models", "verify api mapping"])
        self.assertTrue(ok_payload["ok"])
        self.assertEqual(ok_payload["step_count"], 2)

    def test_send_and_read_message_updates_unread_state(self) -> None:
        sent = self.client_a.send_message("worker-b", body="Need db field nullability.")
        self.assertTrue(sent["ok"])

        state_before = self.hub.worker_state("worker-b")
        self.assertEqual(state_before["unread_count"], 1)
        self.assertTrue(state_before["inbox_dirty"])

        inbox = self.client_b.read_inbox(limit=10, only_unread=True)
        self.assertTrue(inbox["ok"])
        self.assertEqual(len(inbox["messages"]), 1)
        self.assertIn("nullability", inbox["messages"][0]["body"])

        state_after = self.hub.worker_state("worker-b")
        self.assertEqual(state_after["unread_count"], 0)
        self.assertFalse(state_after["inbox_dirty"])

    def test_scope_claim_conflict_and_release(self) -> None:
        claimed = self.client_a.claim_scope(scope="llc/service/api.py")
        self.assertTrue(claimed["ok"])

        conflict = self.client_b.claim_scope(scope="llc/service/api.py")
        self.assertFalse(conflict["ok"])

        released = self.client_a.release_scope(scope="llc/service/api.py")
        self.assertTrue(released["ok"])
        self.assertTrue(released["released"])

        claimed_after_release = self.client_b.claim_scope(scope="llc/service/api.py")
        self.assertTrue(claimed_after_release["ok"])

    def test_worker_coordination_tools_are_not_exposed_to_orchestrator(self) -> None:
        settings = Settings()
        coordination_tools = {
            "SubmitTaskPlan",
            "SendPeerMessage",
            "HasInboxMessages",
            "ReadInbox",
            "WaitForPeerMessage",
            "PostSharedNote",
            "ReadSharedNotes",
            "ClaimScope",
            "ReleaseScope",
            "ListMyClaims",
            "GetCoordinationState",
        }

        orchestrator_names = {
            tool.name
            for tool in collect_tools(
                settings,
                role="orchestrator",
                subagent_runtime=object(),
                subagent_coordination=None,
            )
        }
        default_names = {tool.name for tool in collect_tools(settings, role="default")}
        subagent_names = {
            tool.name
            for tool in collect_tools(
                settings,
                role="subagent",
                subagent_coordination=self.client_a,
            )
        }

        self.assertFalse(coordination_tools & orchestrator_names)
        self.assertFalse(coordination_tools & default_names)
        self.assertTrue(coordination_tools.issubset(subagent_names))


if __name__ == "__main__":
    unittest.main()
