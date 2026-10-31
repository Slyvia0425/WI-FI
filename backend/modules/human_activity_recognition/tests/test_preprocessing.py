from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from backend.modules.human_activity_recognition.errors import DataValidationError, SplitLeakageError
from backend.modules.human_activity_recognition.preprocessing import (
    FeatureStats,
    SampleRecord,
    assert_no_split_leakage,
    canonical_activity,
    check_quality,
    orient_window,
    stable_sample_id,
)

from ._helpers import sample_window


class PreprocessingTest(unittest.TestCase):
    def test_stable_id_is_deterministic_and_unique(self):
        self.assertEqual(stable_sample_id("a", 1), stable_sample_id("a", 1))
        self.assertNotEqual(stable_sample_id("a", 1), stable_sample_id("a", 2))

    def test_aliases_map_to_canonical_ids(self):
        self.assertEqual(canonical_activity("Running").value, "running")
        self.assertEqual(canonical_activity("wavingHand").value, "wavinghand")
        with self.assertRaises(DataValidationError):
            canonical_activity("unknown-activity")

    def test_quality_and_orientation_boundaries(self):
        good = check_quality(
            sample_window(feature_size=8, frames=32), expected_feature_size=8, min_frames=20, max_abs_amplitude=1000
        )
        self.assertTrue(good.ok)
        short = check_quality(
            sample_window(feature_size=8, frames=4), expected_feature_size=8, min_frames=20, max_abs_amplitude=1000
        )
        self.assertFalse(short.ok)
        transposed = orient_window(sample_window(feature_size=32, frames=8), expected_feature_size=8)
        self.assertEqual(transposed.shape[0], 8)
        padded = orient_window(sample_window(feature_size=56, frames=20)[..., None], expected_feature_size=232)
        self.assertEqual(padded.shape, (232, 20))
        self.assertEqual(float(padded[56:].sum()), 0.0)

    def test_normalisation_uses_fit_input_only(self):
        first = sample_window(feature_size=4, frames=16, seed=1, scale=2.0)
        second = sample_window(feature_size=4, frames=16, seed=2, scale=3.0)
        stats = FeatureStats.fit([first])
        transformed = stats.transform(first)
        self.assertLess(float(np.max(np.abs(transformed))), 6.0)
        self.assertNotEqual(stats.mean, FeatureStats.fit([second]).mean)

    def test_window_normalisation_is_session_invariant(self):
        window = sample_window(feature_size=4, frames=16, seed=3)
        shifted = window * 7.5 + 120.0
        stats = FeatureStats.fit([window], mode="window_zscore")
        left = stats.transform(window)
        right = stats.transform(shifted)
        self.assertEqual(stats.mode, "window_zscore")
        np.testing.assert_allclose(left, right, rtol=1e-5, atol=1e-5)
        self.assertLess(float(np.max(np.abs(left.mean(axis=1)))), 1e-5)
        self.assertLess(float(np.max(np.abs(left.std(axis=1) - 1.0))), 1e-5)

    def test_split_leakage_detected(self):
        records = [
            SampleRecord("a", Path("a.h5"), "s1", "u1", "e1", "d1", "jumping", "train"),
            SampleRecord("b", Path("b.h5"), "s1", "u1", "e1", "d1", "running", "test"),
        ]
        with self.assertRaises(SplitLeakageError):
            assert_no_split_leakage(records, group_fields=("session_id",))

    def test_user_overlap_is_allowed_for_in_distribution_split(self):
        records = [
            SampleRecord("a", Path("a.h5"), "s1", "u1", "e1", "d1", "jumping", "train"),
            SampleRecord("b", Path("b.h5"), "s2", "u1", "e1", "d1", "running", "test"),
        ]
        assert_no_split_leakage(records, group_fields=("session_id",))

    def test_empty_quality_is_invalid(self):
        report = check_quality(np.array([]), expected_feature_size=8, min_frames=20, max_abs_amplitude=1000)
        self.assertFalse(report.ok)
