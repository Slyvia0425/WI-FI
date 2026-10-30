import unittest

import numpy as np

from backend.modules.breathing_detection.preprocessing.respiratory_rate import estimate_rates, reference_rate
from backend.modules.breathing_detection.evaluation.evaluate_respiratory_rate import metrics


class RespiratoryRateTest(unittest.TestCase):
    def test_known_rate_with_drift_and_varying_subcarrier_scale(self):
        for duration in (10, 30):
            t = np.arange(duration * 30) / 30
            signal = np.sin(2 * np.pi * .3 * t)
            signals = np.asarray([scale * signal + .1 * t + 10 for scale in (1, 2, 4, 8)])
            for estimate in estimate_rates(signals[None]).values():
                self.assertLess(abs(estimate - 18), 1)
            self.assertLess(abs(reference_rate(signal) - 18), 1)

    def test_flat_signals_rejected(self):
        with self.assertRaises(ValueError):
            estimate_rates(np.ones((1, 5, 300)))

    def test_quality_exclusion_and_record_weighting(self):
        rows = [dict(recording_id=record, assessable=valid, reference_bpm=12,
                     estimates={'mean_spectrum': estimate})
                for record, valid, estimate in [('a', True, 12), ('a', True, 12), ('b', True, 18), ('c', False, 40)]]
        result = metrics(rows, 'mean_spectrum')
        self.assertEqual(result['coverage'], .75)
        self.assertEqual(result['mae_bpm'], 2)
        self.assertEqual(result['record_macro_mae_bpm'], 3)
