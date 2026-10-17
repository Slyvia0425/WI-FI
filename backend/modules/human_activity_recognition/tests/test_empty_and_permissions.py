from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.modules.human_activity_recognition.errors import AuthorizationError, DataNotFoundError
from backend.modules.human_activity_recognition.experiments import load_dataset_split
from backend.modules.human_activity_recognition.service import build_policy

from ._helpers import config_for


class EmptyAndPermissionTest(unittest.TestCase):
    def test_missing_dataset_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = config_for(Path(tmp))
            policy = build_policy(config)
            with self.assertRaises(DataNotFoundError):
                load_dataset_split(config, policy)

    def test_viewer_cannot_read_raw_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = config_for(Path(tmp))
            policy = build_policy(config)
            with self.assertRaises(AuthorizationError):
                policy.authorize_read("viewer", config.paths.data_roots[0] / "HumanActivityRecognition" / "raw.h5")
