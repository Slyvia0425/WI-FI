import unittest

import numpy as np

from backend.modules.breathing_detection.inference.periodic_risk import (
    PeriodicInterferenceScorer,
)
from backend.modules.breathing_detection.inference.temporal_risk import (
    TemporalInstabilityTracker,
)
from backend.modules.breathing_detection.preprocessing.features import FEATURE_NAMES
from backend.modules.breathing_detection.training.train_periodic_gate import (
    select_rejection_threshold,
)


class TemporalInstabilityTest(unittest.TestCase):
    def test_requires_history_and_stays_low_for_stable_confident_windows(self):
        tracker = TemporalInstabilityTracker(0.1, window_size=5, min_history=3)
        self.assertIsNone(tracker.update(0.9))
        self.assertIsNone(tracker.update(0.85))
        self.assertLess(tracker.update(0.95), 0.2)

    def test_alternating_decisions_are_unstable(self):
        tracker = TemporalInstabilityTracker(0.1, window_size=4, min_history=3)
        tracker.update(0.9)
        tracker.update(0.01)
        self.assertEqual(tracker.update(0.8), 1.0)

    def test_repeated_boundary_probabilities_are_ambiguous(self):
        tracker = TemporalInstabilityTracker(0.1, window_size=3, min_history=3)
        tracker.update(0.1)
        tracker.update(0.1)
        self.assertEqual(tracker.update(0.1), 1.0)
        tracker.reset()
        self.assertIsNone(tracker.update(0.9))


class PeriodicRiskTest(unittest.TestCase):
    def test_serialized_scorer_validates_schema_and_scores(self):
        count = len(FEATURE_NAMES)
        payload = {
            "feature_names": list(FEATURE_NAMES),
            "mean": [0.0] * count,
            "scale": [1.0] * count,
            "coefficients": [1.0] + [0.0] * (count - 1),
            "intercept": 0.0,
            "threshold": 0.7,
        }
        scorer = PeriodicInterferenceScorer.from_dict(payload)
        scores = scorer.score(np.asarray([[0.0] * count, [2.0] + [0.0] * (count - 1)]))
        np.testing.assert_allclose(scores, [0.5, 1.0 / (1.0 + np.exp(-2.0))])
        self.assertEqual(scorer.is_interference(np.zeros(count)).item(), False)

    def test_validation_threshold_honours_correct_rejection_budget(self):
        labels = np.array([0, 0, 1, 1, 1])
        base_positive = np.ones(5, dtype=bool)
        risks = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
        threshold, metrics = select_rejection_threshold(
            labels, base_positive, risks, max_correct_rejection=0.0
        )
        self.assertEqual(threshold, 0.8)
        self.assertEqual(metrics["false_positive_coverage"], 1.0)
        self.assertEqual(metrics["correct_breathing_rejection_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
