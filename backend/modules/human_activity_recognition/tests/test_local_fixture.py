from __future__ import annotations

import tempfile
import unittest

from backend.modules.human_activity_recognition.config import load_config
from backend.modules.human_activity_recognition.experiments import load_dataset_split
from backend.modules.human_activity_recognition.service import build_policy
from backend.shared.paths import module_dir


class LocalFixtureTest(unittest.TestCase):
    def test_committed_fixture_loads_only_module_b_subset(self):
        fixture = module_dir("human_activity_recognition") / "fixtures/mini_csi_bench"
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(env={"HAR_DATA_ROOT": str(fixture), "HAR_OUTPUT_ROOT": tmp})
            bundle = load_dataset_split(config, build_policy(config))
        self.assertEqual((len(bundle.train), len(bundle.validation), len(bundle.test)), (8, 4, 4))
        self.assertEqual(bundle.train.windows[0].shape, (232, 24))
