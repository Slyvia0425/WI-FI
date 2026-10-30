"""Checks on the temperature calibration and the threshold trade-off.

    backend/modules/motion_source_recognition/.venv/bin/python -m unittest discover -s backend/modules/motion_source_recognition/tests -t .
"""
import unittest

import numpy as np

from backend.modules.motion_source_recognition.calibration import temper, to_logits
from backend.modules.motion_source_recognition.evaluation.confidence_threshold import class_threshold, decide_per_class, fit_temperature, report_split, trade_off


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class CalibrationTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.n = 6000
        self.logits = rng.normal(size=(self.n, 4)) * 2.0
        true_p = softmax(self.logits)
        self.y = np.array([rng.choice(4, p=p) for p in true_p])   # labels really follow the probabilities of `logits`

    def test_the_temperature_that_undoes_an_over_confident_model(self):
        over_confident = softmax(self.logits * 3.0)               # the model is 3 times too sure
        self.assertAlmostEqual(fit_temperature(to_logits(over_confident), self.y), 3.0, delta=0.35)

    def test_a_well_calibrated_model_keeps_T_near_one(self):
        self.assertAlmostEqual(fit_temperature(to_logits(softmax(self.logits)), self.y), 1.0, delta=0.15)

    def test_the_temperature_never_changes_the_predicted_class(self):
        proba = softmax(self.logits)
        for t in (0.3, 1.0, 5.0):
            np.testing.assert_array_equal(temper(to_logits(proba), t).argmax(axis=1), proba.argmax(axis=1))
            np.testing.assert_allclose(temper(to_logits(proba), t).sum(axis=1), 1.0)


class TradeOffTest(unittest.TestCase):
    def test_rejecting_a_share_removes_that_share_and_the_least_confident(self):
        rng = np.random.default_rng(1)
        proba = softmax(rng.normal(size=(1000, 4)) * 2)
        y = proba.argmax(axis=1)
        y[:100] = (y[:100] + 1) % 4                               # 100 errors
        rows = trade_off(proba, y, shares=[0.0, 0.2])
        self.assertEqual(rows[0]["rejected_share"], 0.0)
        self.assertAlmostEqual(rows[1]["rejected_share"], 0.2, delta=0.01)
        self.assertGreaterEqual(rows[1]["accuracy_kept"], rows[0]["accuracy_kept"] - 1e-9)

    def test_report_counts_rejected_windows(self):
        proba = np.array([[0.9, 0.05, 0.03, 0.02], [0.4, 0.3, 0.2, 0.1], [0.3, 0.6, 0.05, 0.05], [0.5, 0.2, 0.2, 0.1]])
        y = np.array([0, 0, 0, 0])                                # the second is right but unsure, the third wrong and fairly sure
        r = report_split(proba, y, temperature=1.0, threshold=0.45)
        self.assertEqual(r["rejected_share"], 0.25)               # only the window with confidence 0.4 is below 0.45
        self.assertEqual(r["errors_rejected"], 0.0)
        self.assertEqual(r["confident_errors_kept"], 1)


class PerClassThresholdTest(unittest.TestCase):
    def test_lowest_threshold_that_reaches_the_precision_target(self):
        confidence = np.linspace(1.0, 0.1, 100)               # 100 windows from most to least confident
        right = np.array([True] * 60 + [False] * 40)           # the 60 most confident are right, the rest wrong
        t = class_threshold(confidence, right, target=0.9, min_kept=30)
        kept = confidence >= t
        self.assertGreaterEqual(right[kept].mean(), 0.9)
        self.assertGreaterEqual(kept.sum(), 30)
        self.assertLess(t, confidence[59] + 1e-9)               # not stricter than needed...
        self.assertGreater(confidence[kept.sum()], 0) and self.assertLess(right[confidence >= confidence[kept.sum()]].mean(), 0.9)   # ...but one more window would break the target

    def test_none_when_the_target_cannot_be_reached_or_too_few_windows(self):
        confidence, right = np.linspace(1.0, 0.1, 100), np.zeros(100, dtype=bool)
        self.assertIsNone(class_threshold(confidence, right))
        self.assertIsNone(class_threshold(confidence[:10], np.ones(10, dtype=bool)))

    def test_a_class_without_threshold_is_never_claimed_but_leans(self):
        proba = np.array([[0.9, 0.05, 0.03, 0.02], [0.2, 0.1, 0.6, 0.1], [0.3, 0.6, 0.05, 0.05], [0.1, 0.1, 0.1, 0.7]])
        label, lean, confidence = decide_per_class(proba, {0: 0.8, 1: 0.7, 2: None, 3: 0.5})
        self.assertEqual(label.tolist(), [0, -1, -1, 3])       # 2: IRobot never claimed; 3 is below the Pet threshold 0.7
        self.assertEqual(lean.tolist(), [0, 2, 1, 3])


if __name__ == "__main__":
    unittest.main()
