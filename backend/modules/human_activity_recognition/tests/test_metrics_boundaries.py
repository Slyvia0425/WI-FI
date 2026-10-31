from __future__ import annotations

import unittest

from backend.modules.human_activity_recognition.errors import InputValidationError
from backend.modules.human_activity_recognition.evaluation import (
    classification_metrics,
    confusion_matrix,
    expected_calibration_error,
    top_confusions,
)


class MetricsBoundaryTest(unittest.TestCase):
    def test_confusion_and_metrics(self):
        matrix = confusion_matrix([0, 1, 1, 2], [0, 1, 2, 2], 3)
        self.assertEqual(matrix, [[1, 0, 0], [0, 1, 1], [0, 0, 1]])
        metrics = classification_metrics([0, 1, 1, 2], [0, 1, 2, 2], 3)
        self.assertGreater(metrics.macro_f1, 0)
        self.assertEqual(len(metrics.support), 3)

    def test_invalid_label_is_rejected(self):
        with self.assertRaises(InputValidationError):
            confusion_matrix([3], [0], 3)
        with self.assertRaises(InputValidationError):
            classification_metrics([], [], 3)

    def test_calibration_and_top_confusions(self):
        error = expected_calibration_error([0, 1], [[0.9, 0.1], [0.4, 0.6]])
        self.assertGreaterEqual(error, 0)
        self.assertEqual(top_confusions([0, 1], [1, 1], ["jumping", "running"])[0]["true"], "jumping")
