import unittest

from backend.modules.breathing_detection.evaluation.evaluate_rate_reliability import reject, select_rule


class RateReliabilityTest(unittest.TestCase):
    def test_rejection_needs_multiple_weak_signals(self):
        rule = dict(concentration=.2, agreement=.2, peak_ratio=1.3, required_weak=2)
        self.assertFalse(reject(dict(concentration=.1, agreement=.5, peak_ratio=2), rule))
        self.assertTrue(reject(dict(concentration=.1, agreement=.1, peak_ratio=2), rule))
        self.assertFalse(reject(dict(concentration=0, agreement=0, peak_ratio=0), None))

    def test_validation_budget_prevents_rejecting_good_estimates(self):
        rule = dict(concentration=.2, agreement=.2, peak_ratio=1.3, required_weak=2)
        rows = [dict(sample_id=str(i), reference_bpm=12, predictions={'m': 12 if i < 20 else 30}) for i in range(22)]
        features = {r['sample_id']: {'m': dict(concentration=.5, agreement=.5, peak_ratio=3)} for r in rows}
        for row in rows[20:]:
            features[row['sample_id']]['m'] = dict(concentration=.1, agreement=.1, peak_ratio=1)
        selected, counts = select_rule(rows, features, 'm', [rule])
        self.assertEqual(selected, rule)
        self.assertEqual(counts['large_removed'], 2)
        features['0']['m'] = dict(concentration=.1, agreement=.1, peak_ratio=1)
        selected, _ = select_rule(rows, features, 'm', [rule])
        self.assertIsNone(selected)
