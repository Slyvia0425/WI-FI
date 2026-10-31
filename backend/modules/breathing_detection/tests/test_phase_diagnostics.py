import unittest

import numpy as np

from backend.modules.breathing_detection.preprocessing.phase_diagnostics import (
    phase_source_audit, native_phase_traces, phase_windows,
)
from backend.modules.breathing_detection.evaluation.evaluate_common_mode import component_result


class PhaseDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.time = np.arange(1500)/50
        self.targets = np.arange(900)/30
        self.signal = np.sin(2*np.pi*.2*self.time)
        self.h = 10*np.exp(1j*self.signal[:, None]*np.array([.1, .2, .15, .3])[None, :])

    def test_real_complex_values_and_zero_inventory(self):
        h = self.h.copy()
        h[:, 1] = 0
        h[50, 2] = 0
        stats = phase_source_audit(h)
        self.assertEqual(stats['wholly_zero_carriers'], 1)
        self.assertEqual(stats['intermittent_invalid_carriers'], 1)
        self.assertEqual(stats['valid_column_indices'], [0, 3])
        self.assertEqual(stats['valid_adjacent_pairs'], 0)

    def test_removed_column_not_bridged_for_adjacent_pairs(self):
        h = self.h.copy()
        h[:, 1] = 0
        traces, failures, stats = native_phase_traces(h)
        self.assertEqual(stats['valid_pair_left_indices'], [2])
        self.assertEqual(traces['phase_adjacent'].shape, (1500, 1))
        self.assertNotIn('phase_adjacent', failures)

    def test_common_rotation_cancels_from_adjacent_difference(self):
        rotation = 2*np.sin(2*np.pi*.53*self.time)
        a, _, _ = native_phase_traces(self.h)
        b, _, _ = native_phase_traces(self.h*np.exp(1j*rotation[:, None]))
        np.testing.assert_allclose(a['phase_adjacent'], b['phase_adjacent'], atol=1e-12)
        self.assertGreater(np.max(abs(a['phase_raw']-b['phase_raw'])), 1.)

    def test_affine_index_nuisance_projection_invariance_in_regular_case(self):
        position = np.arange(4)
        nuisance = .5*np.sin(self.time)[:, None]+.1*np.cos(self.time)[:, None]*position
        a, _, _ = native_phase_traces(self.h)
        b, _, _ = native_phase_traces(self.h*np.exp(1j*nuisance))
        np.testing.assert_allclose(np.exp(1j*a['phase_index_residual']),
                                   np.exp(1j*b['phase_index_residual']), atol=1e-12)

    def test_common_respiration_can_be_removed_not_just_nuisance(self):
        h = np.tile(10*np.exp(1j*self.signal[:, None]*.3), (1, 4))
        windows, _, _ = phase_windows(self.time, h, self.targets)
        self.assertTrue(component_result(windows['phase_raw'])['available'])
        self.assertFalse(component_result(windows['phase_adjacent'])['available'])
        self.assertFalse(component_result(windows['phase_index_residual'])['available'])

    def test_known_phase_frequency_and_resampled_layout(self):
        windows, failures, _ = phase_windows(self.time, self.h, self.targets)
        self.assertFalse(failures)
        self.assertEqual(windows['phase_raw'].shape, (1, 4, 900))
        self.assertEqual(windows['phase_adjacent'].shape, (1, 3, 900))
        self.assertLess(abs(component_result(windows['phase_raw'])['estimate']['bpm']-12), .5)

    def test_readonly_transforms(self):
        before = self.h.copy()
        phase_windows(self.time, self.h, self.targets)
        np.testing.assert_array_equal(self.h, before)

    def test_undefined_phase_not_interpolated_or_filled(self):
        windows, failures, stats = phase_windows(self.time, np.zeros_like(self.h), self.targets)
        self.assertFalse(windows)
        self.assertEqual(len(failures), 3)
        self.assertEqual(stats['valid_carriers'], 0)

    def test_bad_timestamps_shape_real_inputs_and_extrapolation_rejected(self):
        for time, h, target in ((self.time[::-1], self.h, self.targets),
                                (self.time, self.h[:-1], self.targets),
                                (self.time, self.h.real, self.targets),
                                (self.time, self.h, self.targets-1),
                                (self.time, self.h, self.targets[::-1])):
            with self.assertRaises(ValueError):
                phase_windows(time, h, target)

    def test_nonfinite_carrier_is_excluded_explicitly(self):
        h = self.h.copy()
        h[10, 1] = np.nan+1j
        _, failures, stats = native_phase_traces(h)
        self.assertFalse(stats['all_finite'])
        self.assertEqual(stats['valid_column_indices'], [0, 2, 3])
        self.assertNotIn('phase_raw', failures)

    def test_extra_future_tail_cannot_change_validity_or_phase(self):
        a, _, astats = phase_windows(self.time, self.h, self.targets)
        # The existing last point is a necessary interpolation bracket, not tail.
        time = np.concatenate([self.time, 30+np.arange(100)/50])
        h = np.concatenate([self.h, np.zeros((100, 4), dtype=complex)])
        b, _, bstats = phase_windows(time, h, self.targets)
        self.assertEqual(astats['valid_carriers'], bstats['valid_carriers'])
        for method in a:
            np.testing.assert_array_equal(a[method], b[method])

    def test_unwrap_cannot_recover_true_increment_above_pi(self):
        true_phase = np.arange(20)*4.
        unwrapped = np.unwrap(np.angle(np.exp(1j*true_phase)))
        self.assertLess(np.diff(unwrapped).mean(), 0)
        self.assertGreater(np.diff(true_phase).mean(), 0)


if __name__ == '__main__':
    unittest.main()
