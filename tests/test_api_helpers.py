from __future__ import annotations

import unittest

from llc.commands import CommandRegistry
from llc.config import Settings
from llc.service.api_models import session_summary
from llc.service.api_ws import _worker_snapshot_key
from llc.service.engine import SessionEngine


class ApiHelperTests(unittest.TestCase):
    def test_worker_snapshot_key_is_stable_for_same_payload(self) -> None:
        report = {
            "active_count": 1,
            "max_sub_agents": 5,
            "workers": [{"id": "a", "status": "running", "updated_at": 1.0}],
        }
        self.assertEqual(_worker_snapshot_key(report), _worker_snapshot_key(report))

    def test_session_summary_uses_engine_contract(self) -> None:
        engine = SessionEngine(Settings(), registry=CommandRegistry())
        summary = session_summary(engine)
        self.assertEqual(summary.session_id, engine.session_id)
        self.assertEqual(summary.model_name, engine.model_name)
        self.assertIn(engine.session_id, summary.ws_path)


if __name__ == "__main__":
    unittest.main()
