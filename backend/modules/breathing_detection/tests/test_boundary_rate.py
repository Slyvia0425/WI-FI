import unittest
import numpy as np

from backend.modules.breathing_detection.preprocessing.rate_peaks import (
    conservative_boundary_rate, search_endpoints, local_candidates,
)
from backend.modules.breathing_detection.preprocessing.respiratory_rate import estimate_rates


class BoundaryRateTest(unittest.TestCase):
    def test_interior_estimate_is_never_replaced(self):
        peak = dict(bpm=36., height=1., prominence=1.)
        self.assertEqual(conservative_boundary_rate(12., [peak]), 12.)

    def test_boundary_uses_height_and_preserves_no_peak(self):
        low, high = search_endpoints()
        peaks = [dict(bpm=12., height=1., prominence=.1), dict(bpm=24., height=.8, prominence=.9)]
        self.assertEqual(conservative_boundary_rate(low, peaks), 12.)
        self.assertEqual(conservative_boundary_rate(high, []), high)
        self.assertEqual(conservative_boundary_rate(low, peaks, minimum_prominence=2.), low)

    def test_real_low_rate_signals_are_preserved(self):
        time = np.arange(900) / 30
        for rate in (6., 6.5, 7.5, 9.):
            x = np.tile(np.sin(2 * np.pi * rate / 60 * time), (1, 4, 1))
            baseline = estimate_rates(x)['mean_spectrum']
            corrected = conservative_boundary_rate(baseline, local_candidates(x))
            self.assertLess(abs(corrected - rate), 1.)
