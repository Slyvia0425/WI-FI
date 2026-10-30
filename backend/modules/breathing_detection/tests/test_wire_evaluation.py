import unittest

import numpy as np

from backend.modules.breathing_detection.evaluation.evaluate_wire_breath import (
    false_none_analysis,
    positive_only_metrics,
    sliced_metrics,
)


class WireBreathEvaluationTest(unittest.TestCase):
    def test_positive_only_metrics_exclude_unassessable_windows(self):
        result = positive_only_metrics(
            np.asarray([0.9, 0.2, 0.8, 0.1]),
            np.asarray([True, True, False, True]),
            0.5,
        )
        self.assertEqual(result["windows"], 4)
        self.assertEqual(result["assessable"], 3)
        self.assertEqual(result["breathing_detected"], 1)
        self.assertAlmostEqual(result["breathing_recall"], 1 / 3)

    def test_slice_uses_matching_metadata_rows(self):
        samples = [{"environment": "Quiet"}, {"environment": "Interference"}]
        result = sliced_metrics(
            samples,
            np.asarray([0.9, 0.1]),
            np.asarray([True, True]),
            0.5,
            "environment",
        )
        self.assertEqual(result["Quiet"]["breathing_recall"], 1.0)
        self.assertEqual(result["Interference"]["breathing_recall"], 0.0)

    def test_false_none_analysis_groups_recordings(self):
        samples = [
            {
                "sample_id": f"sample-{index}",
                "recording_id": "recording-a",
                "participant_id": "01",
                "room": "Room1",
                "posture": "Sitting",
                "environment": "Quiet",
            }
            for index in range(2)
        ]
        result = false_none_analysis(
            samples, np.asarray([0.1, 0.9]), np.asarray([True, True]), 0.5
        )
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["recordings_affected"], 1)
        self.assertEqual(result["by_recording"]["recording-a"], 1)


if __name__ == "__main__":
    unittest.main()
