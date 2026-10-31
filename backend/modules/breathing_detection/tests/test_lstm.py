import unittest

import numpy as np
import torch

from backend.modules.breathing_detection.preprocessing.temporal import (
    TEMPORAL_FEATURE_NAMES,
    TOP_VARIANCE_FEATURES,
    INTERPOLATED_FEATURES,
    extract_interpolated_subcarriers,
    extract_temporal_statistics,
    extract_top_variance_subcarriers,
)
from backend.modules.breathing_detection.training.lstm import LightweightBreathingLSTM
from backend.modules.breathing_detection.evaluation.evaluate_lstm import (
    metrics_by_subcarrier_count,
)


class TemporalStatisticsTest(unittest.TestCase):
    def test_output_is_fixed_for_all_real_subcarrier_counts(self):
        rng = np.random.default_rng(12)
        for subcarriers in (112, 208, 240, 416):
            x = rng.normal(size=(1, subcarriers, 300)).astype(np.float32)
            result = extract_temporal_statistics(x)
            self.assertEqual(result.shape, (300, len(TEMPORAL_FEATURE_NAMES)))
            self.assertTrue(np.isfinite(result).all())

    def test_wrong_layout_is_rejected(self):
        with self.assertRaises(ValueError):
            extract_temporal_statistics(np.zeros((208, 300)))

    def test_top_variance_output_is_fixed_normalized_and_deterministic(self):
        rng = np.random.default_rng(9)
        for subcarriers in (112, 208, 240, 416):
            x = rng.normal(size=(1, subcarriers, 300)).astype(np.float32)
            first = extract_top_variance_subcarriers(x)
            second = extract_top_variance_subcarriers(x)
            self.assertEqual(first.shape, (300, TOP_VARIANCE_FEATURES))
            np.testing.assert_array_equal(first, second)
            np.testing.assert_allclose(first.mean(axis=0), 0.0, atol=1e-5)
            np.testing.assert_allclose(first.std(axis=0), 1.0, atol=1e-5)

    def test_top_variance_rejects_non_finite_and_constant_selection(self):
        bad = np.ones((1, 112, 300), dtype=np.float32)
        with self.assertRaises(ValueError):
            extract_top_variance_subcarriers(bad)
        bad[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            extract_top_variance_subcarriers(bad)

    def test_interpolation_preserves_order_and_fixed_shape(self):
        time = np.arange(300, dtype=np.float32)
        for subcarriers in (112, 208, 240, 416):
            scales = np.linspace(1.0, 2.0, subcarriers, dtype=np.float32)[:, None]
            offsets = np.linspace(-3.0, 3.0, subcarriers, dtype=np.float32)[:, None]
            x = (scales * np.sin(time[None, :] / 20.0) + offsets)[None, :, :]
            result = extract_interpolated_subcarriers(x)
            self.assertEqual(result.shape, (300, INTERPOLATED_FEATURES))
            self.assertTrue(np.isfinite(result).all())
            np.testing.assert_allclose(result.mean(axis=0), 0.0, atol=1e-5)

    def test_interpolation_rejects_invalid_data(self):
        with self.assertRaises(ValueError):
            extract_interpolated_subcarriers(np.ones((1, 112, 300), dtype=np.float32))
        bad = np.ones((1, 112, 300), dtype=np.float32)
        bad[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            extract_interpolated_subcarriers(bad)


class LightweightLSTMTest(unittest.TestCase):
    def test_model_shape_and_gradient(self):
        model = LightweightBreathingLSTM(input_size=len(TEMPORAL_FEATURE_NAMES))
        x = torch.randn(4, 300, len(TEMPORAL_FEATURE_NAMES))
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (4,))
        logits.mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_wrong_feature_size_is_rejected(self):
        model = LightweightBreathingLSTM(input_size=len(TEMPORAL_FEATURE_NAMES))
        with self.assertRaises(ValueError):
            model(torch.randn(2, 300, 3))

    def test_projected_model_shape(self):
        model = LightweightBreathingLSTM(input_size=232, projection_size=32)
        logits = model(torch.randn(2, 300, 232))
        self.assertEqual(tuple(logits.shape), (2,))


class GroupedMetricsTest(unittest.TestCase):
    def test_groups_only_assessable_samples_by_original_subcarrier_count(self):
        labels = np.array([0, 1, 0, 1])
        probabilities = np.array([0.1, 0.9, 0.8, 0.7])
        weights = np.array([True, True, False, True])
        counts = np.array([112, 112, 208, 208])
        result = metrics_by_subcarrier_count(
            labels, probabilities, weights, counts, threshold=0.5
        )
        self.assertEqual(result["112"]["count"], 2)
        self.assertEqual(result["208"]["count"], 1)
        self.assertEqual(result["112"]["accuracy"], 1.0)
        self.assertEqual(result["208"]["class_counts"], {"empty": 0, "sleep": 1})
        self.assertIsNone(result["208"]["macro_f1"])
        self.assertIsNone(result["208"]["empty_room_false_positive_rate"])


if __name__ == "__main__":
    unittest.main()
