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
        self.assertEqual(canonical_activity("Standing").value, "standing")
        self.assertEqual(canonical_activity("walk").value, "walking")
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

    def test_normalisation_uses_fit_input_only(self):
        first = sample_window(feature_size=4, frames=16, seed=1, scale=2.0)
        second = sample_window(feature_size=4, frames=16, seed=2, scale=3.0)
        stats = FeatureStats.fit([first])
        transformed = stats.transform(first)
        self.assertLess(float(np.max(np.abs(transformed))), 6.0)
        self.assertNotEqual(stats.mean, FeatureStats.fit([second]).mean)

    def test_split_leakage_detected(self):
        records = [
            SampleRecord("a", Path("a.h5"), "s1", "u1", "e1", "d1", "standing", "train"),
            SampleRecord("b", Path("b.h5"), "s1", "u1", "e1", "d1", "sitting", "test"),
        ]
        with self.assertRaises(SplitLeakageError):
            assert_no_split_leakage(records)

    def test_empty_quality_is_invalid(self):
        report = check_quality(np.array([]), expected_feature_size=8, min_frames=20, max_abs_amplitude=1000)
        self.assertFalse(report.ok)
