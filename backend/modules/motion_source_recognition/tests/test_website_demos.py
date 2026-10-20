"""Checks on the demonstrations shown on the website: the made-up damaged windows, the MiniRocket-style kernel and the feature names.

    backend/modules/motion_source_recognition/.venv/bin/python -m unittest discover -s backend/modules/motion_source_recognition/tests -t .
"""
import unittest

import numpy as np

from backend.modules.motion_source_recognition.evaluation.export_replay import made_up_windows
from backend.modules.motion_source_recognition.evaluation.feature_demo import convolve, kernel_weights
from backend.modules.motion_source_recognition.evaluation.model_anatomy import friendly
from backend.modules.motion_source_recognition.inference import check_quality


class MadeUpWindowsTest(unittest.TestCase):
    def test_three_degraded_and_three_invalid_windows(self):
        base = np.random.default_rng(0).uniform(1, 4, size=(232, 500)).astype(np.float32)
        levels = [check_quality(w)["level"] for _, w in made_up_windows(base)]
        self.assertEqual(levels, ["degraded"] * 3 + ["invalid"] * 3)
        self.assertEqual(check_quality(base)["level"], "ok")   # the real window itself is fine, and is not changed

    def test_the_reasons_name_the_damage(self):
        base = np.random.default_rng(0).uniform(1, 4, size=(232, 500)).astype(np.float32)
        reasons = [check_quality(w)["reasons"][0] for _, w in made_up_windows(base)]
        self.assertEqual(reasons, ["zero_channels:8", "constant_channels:1", "value_out_of_range", "non_finite_values", "wrong_shape", "wrong_shape"])


class KernelTest(unittest.TestCase):
    def test_a_kernel_has_three_plus_weights_and_sums_to_zero(self):
        w = kernel_weights((0, 4, 8))
        self.assertEqual(len(w), 9)
        self.assertEqual(int((w > 0).sum()), 3)
        self.assertEqual(w.sum(), 0)

    def test_only_positions_inside_the_window_are_scored(self):
        x = np.random.default_rng(0).normal(size=500)
        for dilation in (1, 4, 16):
            self.assertEqual(len(convolve(x, kernel_weights((0, 1, 2)), dilation)), 500 - 8 * dilation)

    def test_a_constant_level_gives_no_match_score(self):
        np.testing.assert_allclose(convolve(np.full(500, 3.0), kernel_weights((2, 4, 6)), 4), 0, atol=1e-9)


class NamesTest(unittest.TestCase):
    def test_feature_names_are_readable(self):
        self.assertEqual(friendly("low_band_energy_ch132"), "low band energy, channel 132")
        self.assertEqual(friendly("physical thing"), "physical thing")


if __name__ == "__main__":
    unittest.main()
