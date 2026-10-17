from __future__ import annotations

import time
import unittest

import torch

from backend.modules.human_activity_recognition.training import ModelSpec, build_model


class PerformanceBoundaryTest(unittest.TestCase):
    def test_small_batch_inference_budget_is_bounded(self):
        model = build_model(
            ModelSpec(name="cnn_1d", feature_size=32, num_classes=4, hidden_dim=32, dropout=0.0, heads=4, layers=1)
        )
        windows = torch.zeros(8, 32, 128)
        started = time.perf_counter()
        with torch.no_grad():
            for _ in range(3):
                model(windows)
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 5.0)
