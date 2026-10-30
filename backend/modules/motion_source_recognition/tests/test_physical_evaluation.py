"""Checks on the rolling (dynamics) features and the run numbers of the physical route.

    backend/modules/motion_source_recognition/.venv/bin/python -m unittest discover -s backend/modules/motion_source_recognition/tests -t .
"""
import unittest

import numpy as np
import pandas as pd

from backend.modules.motion_source_recognition.preprocessing.physical import dynamic_features, rolling as _rolling, run_ids


class RollingTest(unittest.TestCase):
    def test_only_past_windows_are_used(self):
        values = np.array([1.0, 1.0, 1.0, 5.0, 1.0, 1.0])
        a = _rolling(values, np.zeros(6), np.arange(6), 3)
        b = _rolling(np.array([1.0, 1.0, 1.0, 9.0, 9.0, 9.0]), np.zeros(6), np.arange(6), 3)   # the future differs
        np.testing.assert_allclose(a.iloc[:3].to_numpy(), b.iloc[:3].to_numpy())

    def test_mean_restarts_in_every_recording(self):
        out = _rolling(np.arange(6, dtype=float), np.array([0, 0, 0, 1, 1, 1]), np.arange(6), 2)
        self.assertEqual(out["mean"].tolist(), [0.0, 0.5, 1.5, 3.0, 3.5, 4.5])

    def test_row_order_does_not_matter(self):
        v = np.random.default_rng(0).uniform(0.5, 1.5, 8)
        order = np.random.default_rng(1).permutation(8)
        base = _rolling(v, np.zeros(8), np.arange(8), 3)
        shuffled = _rolling(v[order], np.zeros(8), np.arange(8)[order], 3)
        np.testing.assert_allclose(base.to_numpy()[order], shuffled.to_numpy())

    def test_change_is_the_mean_absolute_difference_over_the_mean(self):
        out = _rolling(np.array([1.0, 3.0, 1.0]), np.zeros(3), np.arange(3), 3)
        self.assertAlmostEqual(out["change"].iloc[2], 2.0 / (5.0 / 3.0), places=6)   # |diffs| 2 and 2, mean 2; mean of values 5/3

    def test_runs_break_at_a_jump_in_the_session_numbers(self):
        session = np.array([20, 21, 22, 1000, 1001, 5, 6])   # unsorted on purpose: 5, 6 | 20, 21, 22 | 1000, 1001 (a jump larger than 5 ends a run)
        ids = run_ids(session)
        self.assertEqual(len(set(ids)), 3)
        self.assertEqual(ids[0], ids[1])
        self.assertEqual(ids[3], ids[4])
        self.assertNotEqual(ids[2], ids[3])

    def test_dynamic_features_columns(self):
        table = pd.DataFrame({"std": [1.0, 2.0, 1.0], "spec_share_0.5-1": [0.1, 0.2, 0.3], "spec_peak_to_mean": [5.0, 6.0, 7.0]})
        out = dynamic_features(table, np.zeros(3), np.arange(3))
        self.assertEqual(list(out.columns), ["dyn_std_cv", "dyn_std_change", "dyn_share_slow", "dyn_peak_to_mean"])
        table["pca_share_1"], table["ac_peak"], table["permutation_entropy"] = 0.2, 0.5, 0.9
        self.assertEqual(len(dynamic_features(table, np.zeros(3), np.arange(3)).columns), 9)


if __name__ == "__main__":
    unittest.main()
