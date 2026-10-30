import unittest

import numpy as np
from scipy.signal import detrend

from backend.modules.breathing_detection.preprocessing.common_mode import decompose_common_mode
from backend.modules.breathing_detection.evaluation.evaluate_common_mode import component_result, describe_peaks
from backend.modules.breathing_detection.evaluation.evaluate_spike_preprocessing import select_preprocessing


class CommonModeTests(unittest.TestCase):
    def setUp(self):
        self.t = np.arange(900)/30
        self.a = np.sin(2*np.pi*.2*self.t)
        self.b = np.sin(2*np.pi*.4*self.t)

    def test_exact_reconstruction_and_zero_carrier_mean(self):
        x = np.array([self.a+.3*self.b, self.a-.3*self.b, .5*self.a+self.b])[None]
        common, relative, stats = decompose_common_mode(x)
        centered = detrend(x[0], axis=1)
        z = centered/centered.std(axis=1, keepdims=True)
        np.testing.assert_allclose(common[0]+relative[0], z, atol=1e-14)
        np.testing.assert_allclose(relative[0].mean(axis=0), 0., atol=1e-14)
        self.assertLess(stats['energy_sum_error'], 1e-14)

    def test_identical_traces_have_no_relative_component(self):
        x = np.tile(self.a, (1, 4, 1))
        common, relative, stats = decompose_common_mode(x)
        self.assertAlmostEqual(stats['common_energy_fraction'], 1.)
        self.assertLess(stats['relative_energy_fraction'], 1e-25)
        self.assertTrue(component_result(common)['available'])
        self.assertFalse(component_result(relative)['available'])

    def test_opposite_traces_have_no_common_component(self):
        common, relative, stats = decompose_common_mode(np.array([self.a, -self.a])[None])
        self.assertAlmostEqual(stats['relative_energy_fraction'], 1.)
        self.assertFalse(component_result(common)['available'])
        self.assertTrue(component_result(relative)['available'])

    def test_detrending_and_positive_carrier_scale_invariance(self):
        x = np.array([self.a+.3*self.b, self.a-.3*self.b])[None]
        changed = x*np.array([2., 5.])[None, :, None]+np.array([10., 20.])[None, :, None]+self.t*.2
        c, r, _ = decompose_common_mode(x)
        cc, rr, _ = decompose_common_mode(changed)
        np.testing.assert_allclose(c, cc, atol=1e-12)
        np.testing.assert_allclose(r, rr, atol=1e-12)

    def test_carrier_permutation_equivariance(self):
        x = np.array([self.a+.2*self.b, self.a-.1*self.b, self.b])[None]
        c, r, _ = decompose_common_mode(x)
        cc, rr, _ = decompose_common_mode(x[:, [2, 0, 1]])
        np.testing.assert_allclose(c, cc, atol=1e-14)
        np.testing.assert_allclose(r[:, [2, 0, 1]], rr, atol=1e-14)

    def test_flat_carriers_explicitly_dropped_and_inputs_unchanged(self):
        x = np.array([self.a, self.b, np.ones(900)])[None]
        before = x.copy()
        _, relative, stats = decompose_common_mode(x)
        self.assertEqual(stats['dropped_carriers'], 1)
        self.assertEqual(relative.shape, (1, 2, 900))
        np.testing.assert_array_equal(x, before)

    def test_invalid_inputs_and_insufficient_valid_carriers(self):
        for x in (np.zeros((1, 2, 900)), np.ones((1, 2, 899)), np.ones((2, 2, 900)),
                  np.full((1, 2, 900), np.nan), np.array([self.a])[None],
                  np.array([self.a, np.ones(900)])[None]):
            with self.assertRaises(ValueError):
                decompose_common_mode(x)

    def test_tiny_residual_can_have_strong_normalized_spectral_peak(self):
        x = np.array([self.a+1e-5*self.b, self.a-1e-5*self.b])[None]
        _, relative, stats = decompose_common_mode(x)
        self.assertLess(stats['relative_energy_fraction'], 1e-8)
        result = component_result(relative)
        self.assertTrue(result['available'])
        self.assertLess(abs(result['estimate']['bpm']-24), .5)

    def test_reference_near_diagnostics_do_not_choose_predictions(self):
        x = np.tile(self.a, (1, 2, 1))
        result = component_result(x)
        rate = result['estimate']['bpm']
        self.assertTrue(describe_peaks(result, 12.)['reference_near_peak'])
        # Hann-window sidelobes can form tiny genuine local maxima elsewhere.
        self.assertNotEqual(describe_peaks(result, 30.)['best_near_rank'], 1)
        self.assertEqual(result['estimate']['bpm'], rate)
        self.assertFalse(describe_peaks({'available': False}, 12.)['available'])

    def test_missing_component_is_not_silently_baseline(self):
        rows = [dict(recording_id='r', reference_bpm=12., baseline_error=0.,
                     predictions={'baseline': 12., 'common': None, 'relative': 12.})]
        chosen, metrics = select_preprocessing(rows, ['baseline', 'common', 'relative'])
        self.assertEqual(chosen, 'baseline')
        self.assertEqual(metrics['common']['retained'], 0)
        self.assertEqual(metrics['common']['good_rejected'], 1)


if __name__ == '__main__':
    unittest.main()
