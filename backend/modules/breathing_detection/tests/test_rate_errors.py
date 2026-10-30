import unittest

from backend.modules.breathing_detection.evaluation.analyze_rate_errors import classify, select_gate


class RateErrorAnalysisTest(unittest.TestCase):
    def test_harmonic_labels_require_large_error(self):
        self.assertEqual(classify(24, 12), 'near_double')
        self.assertEqual(classify(12, 24), 'near_half')
        self.assertEqual(classify(10, 14), 'not_large')
        self.assertEqual(classify(8, 30), 'other_large')

    def test_gate_uses_good_rejection_budget(self):
        from backend.modules.breathing_detection.evaluation.analyze_rate_errors import FEATURES
        rows = [{'error': 1, 'features': dict.fromkeys(FEATURES, 0.)} for _ in range(20)]
        rows += [{'error': 10, 'features': dict.fromkeys(FEATURES, 1.)} for _ in range(2)]
        gate = select_gate(rows)
        self.assertEqual(gate['validation_large_removed'], 2)
        self.assertEqual(gate['threshold'], 0.)
        self.assertEqual(select_gate(rows[:20])['feature'], 'none')
