import unittest
from copy import deepcopy
import numpy as np

from backend.modules.breathing_detection.preprocessing.peak_support import supported_competitor, RULE_ID
from backend.modules.breathing_detection.evaluation.evaluate_spike_preprocessing import select_preprocessing


def features(rates, peaks):
    return {'carrier_dominant_bpm': rates, 'peaks': [dict(bpm=bpm, height=height) for bpm, height in peaks]}


class PeakSupportTests(unittest.TestCase):
    def test_stronger_supported_competitor_changes_without_reference(self):
        ev = features([12, 12, 12, 24], [(24, 1.), (12, .8)])
        before = deepcopy(ev)
        result = supported_competitor(ev, 24.)
        self.assertEqual(result['bpm'], 12.)
        self.assertTrue(result['changed'])
        self.assertEqual(ev, before)

    def test_tie_and_nearby_peak_preserve_original(self):
        ev = features([12, 24], [(24, 1.), (12, .8)])
        self.assertEqual(supported_competitor(ev, 24.)['bpm'], 24.)
        ev = features([13, 14, 14], [(14, 1.)])
        self.assertFalse(supported_competitor(ev, 12.)['changed'])

    def test_weak_ripple_excluded_and_empty_peaks_preserved(self):
        ev = features([12, 12, 24], [(12, .49), (24, 1.)])
        self.assertEqual(supported_competitor(ev, 24.)['bpm'], 24.)
        self.assertEqual(supported_competitor(features([12], []), 24.)['bpm'], 24.)

    def test_support_recomputed_not_trusted_from_cached_label(self):
        ev = features([24, 24, 12], [(12, .8), (24, 1.)])
        ev['peaks'][0]['carrier_support'] = 1.
        self.assertFalse(supported_competitor(ev, 24.)['changed'])

    def test_majority_can_stably_support_wrong_frequency(self):
        # Known limitation: without reference, support alone cannot certify truth.
        ev = features([24, 24, 24, 12], [(12, 1.), (24, .8)])
        self.assertEqual(supported_competitor(ev, 12.)['bpm'], 24.)

    def test_invalid_or_missing_carrier_evidence_fails_explicitly(self):
        for rates in ([], [np.nan], [60], [[12, 24]]):
            with self.assertRaises(ValueError):
                supported_competitor(features(rates, [(12, 1.)]), 12.)
        with self.assertRaises(ValueError):
            supported_competitor(features([12], [(12, np.nan)]), 12.)
        with self.assertRaises(ValueError):
            supported_competitor(features([12], []), np.nan)

    def test_validation_safety_forces_baseline_despite_one_correction(self):
        rows = [dict(sample_id='a', recording_id='r1', baseline_error=0., reference_bpm=12.,
                     predictions={'baseline': 12., RULE_ID: 24.}),
                dict(sample_id='b', recording_id='r2', baseline_error=12., reference_bpm=12.,
                     predictions={'baseline': 24., RULE_ID: 12.})]
        chosen, scores = select_preprocessing(rows, ['baseline', RULE_ID])
        self.assertEqual(chosen, 'baseline')
        self.assertEqual(scores[RULE_ID]['good_spoiled'], 1)

    def test_tied_validation_prefers_no_change(self):
        rows = [dict(recording_id='r', baseline_error=0., reference_bpm=12.,
                     predictions={'baseline': 12., RULE_ID: 12.})]
        chosen, _ = select_preprocessing(rows, ['baseline', RULE_ID])
        self.assertEqual(chosen, 'baseline')


if __name__ == '__main__':
    unittest.main()
