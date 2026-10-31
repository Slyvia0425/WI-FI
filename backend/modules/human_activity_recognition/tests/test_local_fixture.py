from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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
        self.assertEqual((len(bundle.train), len(bundle.validation), len(bundle.test)), (10, 5, 5))
        self.assertEqual(bundle.train.windows[0].shape, (232, 24))

    def test_fixture_is_learnable_for_both_models(self):
        from dataclasses import replace

        from backend.modules.human_activity_recognition.experiments import run_single_model

        fixture = module_dir("human_activity_recognition") / "fixtures/mini_csi_bench"
        with tempfile.TemporaryDirectory() as tmp:
            base = load_config(env={"HAR_DATA_ROOT": str(fixture), "HAR_OUTPUT_ROOT": tmp})
            config = replace(
                base,
                model=replace(base.model, epochs=60, patience=8, hidden_dim=16, seeds=(42,)),
            )
            bundle = load_dataset_split(config, build_policy(config))
            for model_name in ("cnn_1d", "cnn_transformer"):
                with self.subTest(model=model_name):
                    result = run_single_model(
                        bundle,
                        config,
                        model_name=model_name,
                        seed=42,
                        checkpoint_path=Path(tmp) / model_name / "model.pt",
                        output_dir=Path(tmp) / model_name,
                        mock=True,
                    )
                    self.assertGreaterEqual(result["metrics"]["macro_f1"], 0.9)
