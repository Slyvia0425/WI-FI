"""Checks on the causal smoothing of probabilities over consecutive windows.

    backend/modules/motion_source_recognition/.venv/bin/python -m unittest discover -s backend/modules/motion_source_recognition/tests -t .
"""
import unittest

import numpy as np

from backend.modules.motion_source_recognition.experiments.temporal_smoothing import smooth


class SmoothTest(unittest.TestCase):
    def test_size_one_changes_nothing(self):
        p = np.random.default_rng(0).dirichlet(np.ones(4), size=10).astype(np.float32)
        np.testing.assert_allclose(smooth(p, np.zeros(10), np.arange(10), 1), p, atol=1e-6)

    def test_mean_of_the_last_n_windows_only_uses_the_past(self):
        p = np.array([[1.0, 0], [0, 1], [1, 0], [0, 1]])
        out = smooth(p, np.zeros(4), np.arange(4), 2)
        np.testing.assert_allclose(out[:, 0], [1.0, 0.5, 0.5, 0.5])   # window 0 averages itself; the later windows never leak backwards

    def test_runs_end_at_a_new_group_and_at_a_jump(self):
        p = np.array([[1.0, 0], [1, 0], [0, 1], [0, 1], [1, 0]])
        group = np.array([0, 0, 1, 1, 1])
        session = np.array([10, 11, 20, 21, 1000])
        out = smooth(p, group, session, 24)
        np.testing.assert_allclose(out[:, 0], [1.0, 1.0, 0.0, 0.0, 1.0])

    def test_row_order_does_not_matter(self):
        rng = np.random.default_rng(1)
        p = rng.dirichlet(np.ones(4), size=30)
        group, session = np.repeat([0, 1, 2], 10), np.tile(np.arange(10), 3)
        perm = rng.permutation(30)
        np.testing.assert_allclose(smooth(p[perm], group[perm], session[perm], 4), smooth(p, group, session, 4)[perm], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
