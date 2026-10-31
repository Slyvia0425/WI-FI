from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.modules.human_activity_recognition.experiments import run_single_model, synthetic_split_bundle
from backend.modules.human_activity_recognition.service import HumanActivityService

from ._helpers import sample_window, tiny_config


class ServiceEndToEndTest(unittest.TestCase):
    def test_training_checkpoint_loads_and_predicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = tiny_config(root)
            bundle = synthetic_split_bundle(
                feature_size=config.dataset.expected_feature_size, frames=max(24, config.dataset.min_frames)
            )
            checkpoint = config.paths.output_root / "model.pt"
            result = run_single_model(
                bundle, config, model_name="cnn_1d", seed=7, checkpoint_path=checkpoint, mock=True
            )
            self.assertEqual(result["replay"]["mock"], True)
            service = HumanActivityService(config)
            service.load_checkpoint(checkpoint)
            prediction = service.predict(
                sample_window(
                    feature_size=config.dataset.expected_feature_size, frames=max(24, config.dataset.min_frames)
                )
            )
            self.assertIn(
                prediction.activity.value,
                {"jumping", "running", "seated-breathing", "walking", "wavinghand", "unknown"},
            )
            self.assertGreaterEqual(prediction.confidence, 0.0)
            self.assertLessEqual(prediction.confidence, 1.0)

    def test_service_returns_structured_error_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = HumanActivityService(tiny_config(Path(tmp)))
            payload = service.handle({"action": "predict", "window": [[0.0] * 24 for _ in range(232)]})
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["error"]["code"], "MODEL_UNAVAILABLE")
