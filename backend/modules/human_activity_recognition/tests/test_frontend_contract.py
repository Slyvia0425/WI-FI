from __future__ import annotations

import json
import unittest

from backend.shared.contract import validate_replay
from backend.shared.paths import REPO_ROOT


class FrontendContractTest(unittest.TestCase):
    def test_bundled_replay_matches_contract(self):
        path = REPO_ROOT / "frontend" / "src" / "modules" / "human_activity_recognition" / "data" / "replay.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(validate_replay("human_activity_recognition", payload), [])

    def test_frontend_uses_optional_fields_safely(self):
        source = (REPO_ROOT / "frontend" / "src" / "modules" / "human_activity_recognition" / "index.jsx").read_text(
            encoding="utf-8"
        )
        self.assertIn("f.probabilities", source)
        self.assertIn("f.activity", source)
