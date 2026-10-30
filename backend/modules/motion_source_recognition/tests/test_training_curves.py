"""Checks on the small helpers of the training curves.

    backend/modules/motion_source_recognition/.venv/bin/python -m unittest discover -s backend/modules/motion_source_recognition/tests -t .
"""
import unittest

import numpy as np

from backend.modules.motion_source_recognition.evaluation.training_curves import ITERATIONS_AT, TREES_AT, running_mean


class HelperTest(unittest.TestCase):
    def test_running_mean_averages_the_first_n_arrays(self):
        parts = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])]
        out = running_mean(parts, [1, 3])
        np.testing.assert_allclose(out[1], [1, 0])
        np.testing.assert_allclose(out[3], [2 / 3, 2 / 3])
        self.assertEqual(sorted(out), [1, 3])

    def test_the_last_stage_is_the_saved_size_of_the_models(self):
        self.assertEqual(TREES_AT[-1], 300)
        self.assertEqual(ITERATIONS_AT[-1], 300)


if __name__ == "__main__":
    unittest.main()
