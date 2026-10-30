"""Checks on the single-window inference: the quality levels, the unknown rule, and (when the trained files exist) a whole frame.

    backend/modules/motion_source_recognition/.venv/bin/python -m unittest discover -s backend/modules/motion_source_recognition/tests -t .
"""
import unittest

import numpy as np

from backend.modules.motion_source_recognition.inference import FROZEN, check_quality, decide
from backend.modules.motion_source_recognition.preprocessing.config import CLASSES, PREP_DIR


def window(seed=0):
    return np.random.default_rng(seed).uniform(0.5, 3.5, size=(232, 500)).astype(np.float32)


class QualityTest(unittest.TestCase):
    def test_an_ordinary_window_is_ok_in_both_layouts(self):
        for x in (window(), window()[..., None]):
            self.assertEqual(check_quality(x), {"ok": True, "level": "ok", "reasons": []})

    def test_unusable_windows_are_invalid(self):
        bad = window()
        bad[3, 7] = np.nan
        self.assertEqual(check_quality(bad)["reasons"], ["non_finite_values"])
        self.assertEqual(check_quality(window()[:100])["reasons"], ["wrong_shape"])
        self.assertEqual(check_quality(None)["level"], "invalid")

    def test_zero_constant_and_out_of_range_are_degraded_with_reasons(self):
        x = window()
        x[0] = 0.0
        x[1] = 1.7
        x[2, 0] = 12.0
        q = check_quality(x)
        self.assertEqual((q["ok"], q["level"]), (False, "degraded"))
        self.assertEqual(q["reasons"], ["zero_channels:1", "constant_channels:1", "value_out_of_range"])


class DecideTest(unittest.TestCase):
    def test_unknown_below_the_threshold(self):
        probs = np.array([0.7, 0.2, 0.05, 0.05])
        self.assertEqual(decide(probs, 0.6), ("Human", 0.7))
        self.assertEqual(decide(probs, 0.8), ("unknown", 0.7))
        self.assertEqual(decide(np.array([0.1, 0.1, 0.1, 0.7]), 0.5)[0], "Fan")


@unittest.skipUnless(FROZEN.exists() and (PREP_DIR / "minirocket_split_0.pkl").exists(), "train the model and freeze the threshold first")
class FrameTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from backend.modules.motion_source_recognition.inference import Detector
        cls.detector = Detector(split=0)

    def test_frame_fields_and_probabilities(self):
        frame = self.detector(window(1))
        self.assertEqual(set(frame), {"label", "confidence", "probs", "quality"})
        self.assertEqual(list(frame["probs"]), CLASSES)
        self.assertAlmostEqual(sum(frame["probs"].values()), 1.0, places=2)
        self.assertEqual(frame["confidence"], round(max(frame["probs"].values()), 4))
        self.assertIn(frame["label"], CLASSES + ["unknown"])
        self.assertEqual(frame["label"] == "unknown", frame["confidence"] < self.detector.threshold)

    def test_an_invalid_window_gives_unknown_and_no_probabilities(self):
        x = window()
        x[0, 0] = np.inf
        frame = self.detector(x)
        self.assertEqual((frame["label"], frame["confidence"], frame["probs"]), ("unknown", 0.0, None))

    def test_smoothing_averages_with_the_previous_windows(self):
        x = window(2)
        alone = self.detector.probabilities(x)
        other = np.array([1.0, 0.0, 0.0, 0.0])
        frame = self.detector(x, previous=[other])
        np.testing.assert_allclose(list(frame["probs"].values()), (alone + other) / 2, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
