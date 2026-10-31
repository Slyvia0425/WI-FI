import unittest
from copy import deepcopy

import numpy as np

from backend.modules.breathing_detection.preprocessing.rate_temporal_reliability import (
    temporal_context, candidate_features, reliability_rules, withhold_rate,
)
from backend.modules.breathing_detection.evaluation.evaluate_rate_reliability_stage2 import metrics, budget_pass, select_rule


def signal(bpm=12):
    t = np.arange(900)/30
    return np.sin(2*np.pi*bpm/60*t[None, None, :] + np.arange(5)[None, :, None]*.1)


def rows_and_features():
    rows = [dict(sample_id=str(i), participant_id='01' if i < 100 else '02',
                 recording_id='recordA' if i < 100 else 'recordB', reference_bpm=12,
                 predictions={'m': 12 if i < 140 else 30}) for i in range(141)]
    features = {r['sample_id']: {'m': dict(available=True, temporal_disagreement_count=0,
                    harmonic_competition_ratio=0., harmonic_subwindow_votes=0)} for r in rows}
    return rows, features


class RateReliabilityStage2Test(unittest.TestCase):
    def test_stable_signal_and_genuine_low_rate_are_not_rejected(self):
        for bpm in (6, 9, 12, 18, 30):
            f = candidate_features(temporal_context(signal(bpm)), bpm)
            self.assertEqual(f['temporal_disagreement_count'], 0)
            self.assertTrue(all(not withhold_rate(f, rule) for rule in reliability_rules()))
            self.assertEqual(f['harmonic_subwindow_votes'], 0)

    def test_wrong_half_candidate_has_temporal_and_harmonic_evidence(self):
        f = candidate_features(temporal_context(signal(24)), 12)
        self.assertEqual(f['temporal_disagreement_count'], 3)
        self.assertEqual(f['harmonic_subwindow_votes'], 3)
        self.assertAlmostEqual(f['harmonic_competitor_bpm'], 24, delta=1)
        self.assertGreater(f['harmonic_competition_ratio'], 1)
        self.assertTrue(all(withhold_rate(f, rule) for rule in reliability_rules()))

    def test_stable_dominant_peak_cannot_be_certified_as_physiological(self):
        # If 24 BPM is interference and true breathing is elsewhere, a stable
        # dominant peak alone cannot reveal that; the reference is never input.
        f = candidate_features(temporal_context(signal(24)), 24)
        self.assertEqual(f['temporal_disagreement_count'], 0)
        self.assertTrue(all(not withhold_rate(f, rule) for rule in reliability_rules()))

    def test_conjunction_requires_both_time_and_competitor_evidence(self):
        rule = dict(min_disagreements=2, min_ratio=1., min_harmonic_votes=2)
        f = dict(available=True, temporal_disagreement_count=2, harmonic_competition_ratio=2, harmonic_subwindow_votes=2)
        self.assertTrue(withhold_rate(f, rule))
        self.assertFalse(withhold_rate({**f, 'temporal_disagreement_count': 1}, rule))
        self.assertFalse(withhold_rate({**f, 'harmonic_competition_ratio': .9}, rule))
        self.assertFalse(withhold_rate({**f, 'harmonic_subwindow_votes': 1}, rule))
        self.assertFalse(withhold_rate({'available': False}, rule))
        self.assertFalse(withhold_rate(f, None))

    def test_feature_inputs_are_read_only_and_invalid_signals_raise(self):
        x = signal()
        original = x.copy()
        candidate_features(temporal_context(x), 12)
        np.testing.assert_array_equal(x, original)
        for invalid in (x[..., :899], x[0], np.zeros_like(x), np.full_like(x, np.nan)):
            with self.assertRaises(ValueError):
                temporal_context(invalid)
        with self.assertRaises(ValueError):
            candidate_features(temporal_context(x), 50)
        partially_flat = x.copy()
        partially_flat[..., :600] = 0
        with self.assertRaises(ValueError):
            temporal_context(partially_flat)

    def test_selection_rejects_harm_hidden_by_pooled_average(self):
        rows, features = rows_and_features()
        rule = reliability_rules()[0]
        for sample_id in ('100', '140'):
            features[sample_id]['m']['temporal_disagreement_count'] = 3
        mask = [withhold_rate(features[r['sample_id']]['m'], rule) for r in rows]
        self.assertLess(metrics(rows, 'm', mask)['good_withholding_rate'], .01)
        self.assertFalse(budget_pass(rows, 'm', mask))
        selected, audit = select_rule(rows, features, 'm', [rule])
        self.assertIsNone(selected)
        self.assertFalse(audit[0]['eligible'])

    def test_safe_interception_selected_without_mutating_predictions(self):
        rows, features = rows_and_features()
        original = deepcopy(rows)
        features['140']['m']['temporal_disagreement_count'] = 3
        rule = reliability_rules()[0]
        selected, _ = select_rule(rows, features, 'm', [rule])
        self.assertEqual(selected, rule)
        self.assertEqual(rows, original)
        m = metrics(rows, 'm', [r['sample_id'] == '140' for r in rows])
        self.assertEqual(m['current_good'], 140)
        self.assertEqual(m['large_total'], 1)
        self.assertEqual(m['good_withheld'], 0)
        self.assertEqual(m['large_withheld'], 1)
        self.assertEqual(m['mae_bpm'], 0.)

    def test_no_benefit_prefers_no_gate_and_metrics_use_current_estimator(self):
        rows, features = rows_and_features()
        self.assertIsNone(select_rule(rows, features, 'm', reliability_rules())[0])
        m = metrics(rows, 'm', np.zeros(len(rows), dtype=bool))
        self.assertEqual(m['retained'], 141)
        self.assertEqual(m['large_retained'], 1)
        self.assertEqual(m['good_withholding_rate'], 0)
        self.assertEqual(len(reliability_rules()), 9)
