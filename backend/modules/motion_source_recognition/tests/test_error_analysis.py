"""Checks on the helpers of the error analysis.

    backend/modules/motion_source_recognition/.venv/bin/python -m unittest discover -s backend/modules/motion_source_recognition/tests -t .
"""
import unittest

import numpy as np
import pandas as pd

from backend.modules.motion_source_recognition.training.common import confusion, human_vs_rest
from backend.modules.motion_source_recognition.evaluation.error_analysis import confident_mistakes, room_type_table, row_normalised

# classes: 0 Human, 1 Pet, 2 IRobot, 3 Fan
Y = np.array([0, 0, 0, 0, 1, 1, 2, 2, 3, 3])
CALLED = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 1])


class ErrorAnalysisTest(unittest.TestCase):
    def test_confusion_counts_and_rows_sum_to_one(self):
        m = confusion(Y, CALLED)
        self.assertEqual(m[0].tolist(), [3, 1, 0, 0])
        self.assertEqual(m.sum(), 10)
        np.testing.assert_allclose(row_normalised(m).sum(axis=1), 1.0)

    def test_empty_row_stays_zero(self):
        self.assertEqual(row_normalised(np.zeros((4, 4)))[2].tolist(), [0, 0, 0, 0])

    def test_human_vs_rest_counts_false_alarms_by_class(self):
        r = human_vs_rest(Y, CALLED)
        self.assertEqual((r["precision"], r["recall"]), (round(3 / 7, 3), 0.75))   # called Human: 3 Humans, 1 Pet, 2 IRobot, 1 Fan
        self.assertEqual(r["false_alarms"]["IRobot"]["windows"], 2)
        self.assertEqual(r["false_alarms"]["IRobot"]["share_of_that_class"], 1.0)
        self.assertEqual(r["misses_called"]["Pet"], 1)

    def test_confident_mistakes(self):
        proba = np.full((4, 4), 0.025)
        proba[[0, 1, 2], [0, 1, 1]] = 0.925   # windows 0..2 are confident; 3 is not
        proba[3] = 0.25
        r = confident_mistakes(proba, np.array([0, 0, 1, 0]))   # window 1 is a confident error; window 3 is a doubtful (random) one
        self.assertEqual(r["windows_at_least_0.9"], 0.75)
        self.assertAlmostEqual(r["wrong_among_those"], 1 / 3, places=3)
        self.assertEqual(r["confident_errors_by_true_class"]["Human"], 1)

    def test_room_types(self):
        meta = pd.DataFrame({"environment": ["A", "A", "B", "B"]})
        out = room_type_table(np.array([0, 0, 0, 1]), np.array([0, 0, 0, 0]), meta, {"A": ["Human"], "B": ["Human", "Pet"]})
        self.assertEqual(out["all_windows"]["rooms_with_one_class"], {"windows": 2, "error_rate": 0.0})
        self.assertEqual(out["all_windows"]["rooms_shared_by_several_classes"], {"windows": 2, "error_rate": 0.5})
        out = room_type_table(np.array([0, 0, 0, 3]), np.array([0, 0, 0, 0]), meta, {"A": ["Human"], "B": ["Human", "Fan"]})   # the Fan window is left out of the Human/Pet figure
        self.assertEqual(out["human_and_pet_only"]["rooms_shared_by_several_classes"], {"windows": 1, "error_rate": 0.0})


if __name__ == "__main__":
    unittest.main()
