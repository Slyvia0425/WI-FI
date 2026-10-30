import unittest
from unittest.mock import patch
import numpy as np

from backend.modules.breathing_detection.preprocessing.unified_rate import estimate_unified_rate, ESTIMATOR_ID
from backend.modules.breathing_detection.preprocessing.rate_peaks import conservative_boundary_rate, local_candidates, search_endpoints
from backend.modules.breathing_detection.preprocessing.respiratory_rate import estimate_rates
from backend.modules.breathing_detection.inference.streaming import BreathingStream, StreamConfig

MODULE = 'backend.modules.breathing_detection.preprocessing.unified_rate'


def signal(bpm=18.):
    t = np.arange(900)/30
    return np.sin(2*np.pi*bpm/60*t[None,None,:] + np.arange(4)[None,:,None]*.2)


class UnifiedRateTest(unittest.TestCase):
    def test_interior_unchanged_without_candidate_search(self):
        with patch(MODULE+'.estimate_rates', return_value={'mean_spectrum': 18.}), patch(MODULE+'.local_candidates') as peaks:
            result = estimate_unified_rate(signal())
        peaks.assert_not_called()
        self.assertEqual(result['bpm'], 18.)
        self.assertEqual(result['reason'], 'interior_preserved')
        self.assertFalse(result['correction_applied'])

    def test_exact_endpoints_use_height_without_prominence_cutoff(self):
        peaks = [dict(bpm=12.,height=1.,prominence=.001), dict(bpm=24.,height=.8,prominence=.8)]
        for endpoint in search_endpoints():
            with patch(MODULE+'.estimate_rates', return_value={'mean_spectrum':endpoint}), patch(MODULE+'.local_candidates', return_value=peaks):
                result = estimate_unified_rate(signal())
            self.assertEqual(result['bpm'], 12.)
            self.assertEqual(result['baseline_bpm'], endpoint)
            self.assertTrue(result['correction_applied'])
            self.assertEqual(result['estimator_id'], ESTIMATOR_ID)

    def test_no_local_peak_fallback_is_not_reliability_claim(self):
        low, _ = search_endpoints()
        with patch(MODULE+'.estimate_rates', return_value={'mean_spectrum':low}), patch(MODULE+'.local_candidates', return_value=[]):
            result = estimate_unified_rate(signal())
        self.assertEqual(result['bpm'], low)
        self.assertEqual(result['reason'], 'boundary_no_local_peak')
        self.assertFalse(result['reliability_gate_enabled'])

    def test_near_endpoint_is_not_exact_endpoint(self):
        low, _ = search_endpoints()
        with patch(MODULE+'.estimate_rates', return_value={'mean_spectrum':low+.001}), patch(MODULE+'.local_candidates') as peaks:
            result = estimate_unified_rate(signal())
        peaks.assert_not_called()
        self.assertEqual(result['bpm'], low+.001)

    def test_readonly_low_rate_and_original_fixed_rule_equivalence(self):
        for bpm in (6.,6.5,7.5,9.,12.,18.,30.,40.):
            x = signal(bpm)
            original = x.copy()
            result = estimate_unified_rate(x)
            expected = conservative_boundary_rate(estimate_rates(x)['mean_spectrum'], local_candidates(x), 0.)
            self.assertEqual(result['bpm'], expected)
            self.assertLess(abs(result['bpm']-bpm), 1.)
            np.testing.assert_array_equal(x, original)

    def test_reject_short_window_invalid_rate_or_signal(self):
        x = signal()
        for invalid in (x[0], x[...,:300], x[...,:899], np.zeros_like(x), np.full_like(x,np.nan)):
            with self.assertRaises(ValueError):
                estimate_unified_rate(invalid)
        with self.assertRaises(ValueError):
            estimate_unified_rate(x, 50.)
        with self.assertRaises(ValueError):
            StreamConfig(rate_seconds=20.)
        with self.assertRaises(ValueError):
            StreamConfig(target_rate_hz=50.)

    def test_stream_calls_shared_estimator_only_when_full_and_supported(self):
        t = np.arange(901)/30
        x = np.sin(2*np.pi*.3*t[:,None]+np.arange(4)[None,:]*.2)
        details = dict(bpm=17.,baseline_bpm=6.15234375,estimator_id=ESTIMATOR_ID,
                       correction_applied=True,reason='boundary_corrected',reliability_gate_enabled=False)
        with patch('backend.modules.breathing_detection.inference.streaming.estimate_unified_rate', return_value=details) as estimate:
            result = BreathingStream(lambda sequence: .9).push(t,x,arrival_time=0)
            self.assertTrue(all(r['rate_bpm'] is None for r in result[:-1]))
            self.assertEqual(result[-1]['rate_bpm'], 17.)
            self.assertEqual(result[-1]['rate_details'], details)
            self.assertEqual(result[-1]['rate_estimator_id'], ESTIMATOR_ID)
            estimate.assert_called_once()
        with patch('backend.modules.breathing_detection.inference.streaming.estimate_unified_rate') as estimate:
            result = BreathingStream(lambda sequence: .001).push(t,x,arrival_time=0)
            self.assertIsNone(result[-1]['rate_bpm'])
            estimate.assert_not_called()
