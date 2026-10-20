"""Checks on the pieces of the test scoring that do not need the test rows.

    backend/modules/motion_source_recognition/.venv/bin/python -m unittest discover -s backend/modules/motion_source_recognition/tests -t .
"""
import unittest

import numpy as np

from backend.modules.motion_source_recognition.evaluation.test_scoring import bootstrap, confusion_by_group, macro_f1_present, stratified_draws
from backend.modules.motion_source_recognition.evaluation.baseline_comparison import per_class
from backend.modules.motion_source_recognition.training.common import score


class ConfusionTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.y = rng.integers(0, 4, 400)
        self.predicted = np.where(rng.random(400) < 0.7, self.y, rng.integers(0, 4, 400))
        self.group = rng.integers(0, 12, 400)

    def test_group_counts_add_up_to_the_whole_confusion_matrix(self):
        counts, ids = confusion_by_group(self.y, self.predicted, self.group)
        self.assertEqual(len(ids), len(np.unique(self.group)))
        total = counts.sum(axis=0).reshape(4, 4)
        expected = np.array(score(self.y, np.eye(4)[self.predicted])["confusion_matrix"])
        np.testing.assert_array_equal(total, expected)

    def test_macro_f1_from_the_confusion_matrix_equals_the_score_function(self):
        counts, _ = confusion_by_group(self.y, self.predicted, self.group)
        expected = score(self.y, np.eye(4)[self.predicted])["macro_f1"]
        self.assertAlmostEqual(macro_f1_present(counts.sum(axis=0).reshape(4, 4)), expected, places=3)

    def test_a_class_with_no_true_window_is_left_out(self):
        confusion = np.diag([5.0, 5.0, 5.0, 0.0])
        self.assertEqual(macro_f1_present(confusion), 1.0)

    def test_resampling_all_groups_once_gives_the_original_score(self):
        counts, ids = confusion_by_group(self.y, self.predicted, self.group)
        draws = np.ones((1, len(ids)))
        out = bootstrap({"m": counts}, draws)["m"]
        self.assertAlmostEqual(out["macro_f1"][0], macro_f1_present(counts.sum(axis=0).reshape(4, 4)))
        self.assertAlmostEqual(out["accuracy"][0], (self.y == self.predicted).mean())

    def test_stratified_draws_keep_every_class_and_a_single_group_fixed(self):
        group_class = np.array([0, 0, 0, 1, 1, 2, 3])   # classes 2 and 3 have a single group each
        draws = stratified_draws(group_class, 200, np.random.default_rng(0))
        np.testing.assert_array_equal(draws[:, :3].sum(axis=1), 3)
        np.testing.assert_array_equal(draws[:, 3:5].sum(axis=1), 2)
        np.testing.assert_array_equal(draws[:, 5], 1)
        np.testing.assert_array_equal(draws[:, 6], 1)


    def test_per_class_recall_and_f1_from_a_confusion_matrix(self):
        confusion = np.array([[8.0, 2, 0, 0], [1, 9, 0, 0], [5, 5, 0, 0], [0, 0, 0, 0]])   # IRobot never found; Fan absent
        r = per_class(confusion)
        self.assertEqual(r["Human"]["recall"], 0.8)
        self.assertEqual(r["IRobot"]["recall"], 0.0)
        self.assertIsNone(r["Fan"]["recall"])
        self.assertAlmostEqual(r["Human"]["f1"], 2 * 8 / (10 + 14), places=3)


if __name__ == "__main__":
    unittest.main()
