import unittest
from copy import deepcopy

from backend.modules.breathing_detection.preprocessing.three_route import match_peak, three_route_features


def component(bpm, peaks=None):
    if peaks is None:
        peaks = [dict(bpm=bpm, height=1., prominence=.8)]
    return {'available': True, 'estimate': {'bpm': bpm}, 'evidence': {'peaks': peaks}}


class ThreeRouteTests(unittest.TestCase):
    def test_agreement_and_proposal_use_fixed_tolerance(self):
        f = three_route_features({'baseline': component(24), 'common': component(12), 'relative': component(13)})
        self.assertTrue(f['proposes_change'])
        self.assertTrue(f['common_relative_agree'])
        self.assertFalse(f['baseline_common_agree'])
        self.assertEqual(f['baseline_relative_distance_bpm'], 11)

    def test_exact_tolerance_is_agreement_not_proposal(self):
        f = three_route_features({'baseline': component(12), 'common': component(14), 'relative': component(14)})
        self.assertTrue(f['baseline_common_agree'])
        self.assertFalse(f['proposes_change'])

    def test_missing_component_explicit_and_no_invented_support(self):
        f = three_route_features({'baseline': component(12), 'common': {'available': False}, 'relative': {'available': False}})
        self.assertFalse(f['all_available'])
        self.assertFalse(f['proposes_change'])
        self.assertIsNone(f['baseline_common_distance_bpm'])
        self.assertIsNone(f['matched_peaks']['common']['baseline'])

    def test_peak_matching_height_then_distance_then_frequency(self):
        peaks = [dict(bpm=11., height=.7, prominence=.5), dict(bpm=13., height=.8, prominence=.6)]
        self.assertEqual(match_peak(peaks, 12.)['rank'], 2)
        peaks[0]['height'] = .8
        self.assertEqual(match_peak(peaks, 12.)['bpm'], 11.)
        self.assertIsNone(match_peak(peaks, 30.))

    def test_features_are_readonly_and_ignore_unrelated_reference_fields(self):
        inputs = {name: component(12) for name in ('baseline', 'common', 'relative')}
        before = deepcopy(inputs)
        first = three_route_features(inputs)
        self.assertEqual(inputs, before)
        for value in inputs.values():
            value['reference_bpm'] = 30.
            value['reference_near_peak'] = False
        self.assertEqual(three_route_features(inputs), first)

    def test_same_agreement_pattern_is_not_correctness_label(self):
        a = three_route_features({'baseline': component(23.3), 'common': component(11.9), 'relative': component(12.3)})
        b = three_route_features({'baseline': component(23.7), 'common': component(12.3), 'relative': component(11.)})
        for key in ('proposes_change', 'baseline_common_agree', 'common_relative_agree'):
            self.assertEqual(a[key], b[key])
        self.assertNotIn('reliable', a)
        self.assertNotIn('prediction', a)

    def test_invalid_available_rate_or_peak_evidence_rejected(self):
        for rate in (float('nan'), 50.):
            with self.assertRaises(ValueError):
                three_route_features({'baseline': component(rate), 'common': component(12), 'relative': component(12)})
        with self.assertRaises(ValueError):
            three_route_features({'baseline': {'available': False}, 'common': component(12), 'relative': component(12)})
        for peak in (dict(bpm=12., height=float('nan'), prominence=.8), dict(bpm=12., height=2., prominence=.8)):
            with self.assertRaises(ValueError):
                three_route_features({'baseline': component(12, [peak]), 'common': component(12), 'relative': component(12)})


if __name__ == '__main__':
    unittest.main()
