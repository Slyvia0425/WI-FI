"""Checks on the pieces of the latency measurement that need no data.

    backend/modules/motion_source_recognition/.venv/bin/python -m unittest discover -s backend/modules/motion_source_recognition/tests -t .
"""
import unittest

import numpy as np
import pandas as pd

from backend.modules.motion_source_recognition.evaluation.latency import check_dynamics, dynamics_row, summary
from backend.modules.motion_source_recognition.preprocessing.physical import dynamic_features


class StreamingDynamicsTest(unittest.TestCase):
    def test_streaming_dynamics_equal_the_training_code(self):
        check_dynamics([])   # raises when they differ

    def test_also_at_the_start_of_a_run(self):
        rng = np.random.default_rng(3)
        keys = ["std", "spec_share_0.5-1", "spec_peak_to_mean", "pca_share_1", "ac_peak", "permutation_entropy"]
        table = pd.DataFrame({k: rng.uniform(0.2, 2.0, 8) for k in keys})
        reference = dynamic_features(table, np.zeros(8, dtype=int), np.arange(8))
        for n in (1, 2, 5, 8):   # the run has only n windows so far
            history = {k: list(table[k].to_numpy()[:n]) for k in keys}
            np.testing.assert_allclose(dynamics_row(history), reference.iloc[n - 1].to_numpy(), rtol=1e-6, err_msg=f"{n} windows")

    def test_summary(self):
        s = summary([1_000_000, 2_000_000, 3_000_000])
        self.assertEqual((s["median_ms"], s["mean_ms"]), (2.0, 2.0))


if __name__ == "__main__":
    unittest.main()
