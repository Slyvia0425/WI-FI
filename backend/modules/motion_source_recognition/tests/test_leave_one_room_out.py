"""Checks on the leave-one-room-out summary.

    backend/modules/motion_source_recognition/.venv/bin/python -m unittest discover -s backend/modules/motion_source_recognition/tests -t .
"""
import unittest

import numpy as np

from backend.modules.motion_source_recognition.evaluation.leave_one_room_out import pooled


class PooledTest(unittest.TestCase):
    def test_pooled_scores_come_from_the_summed_confusion_matrices(self):
        a = {"confusion_matrix": np.diag([8, 0, 0, 0]).tolist()}
        b = {"confusion_matrix": [[0, 0, 0, 0], [2, 6, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}   # 2 Pets called Human, 6 Pets right
        r = pooled({"A": a, "B": b})
        self.assertEqual(r["windows"], 16)
        self.assertEqual(r["accuracy"], round(14 / 16, 3))
        self.assertEqual(r["recall"]["Pet"], 0.75)
        self.assertEqual(r["human_vs_rest"]["false_alarms"]["Pet"]["windows"], 2)


if __name__ == "__main__":
    unittest.main()
