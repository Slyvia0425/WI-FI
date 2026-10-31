import unittest
import numpy as np

from backend.modules.breathing_detection.preprocessing.rate_peaks import choose_peak, local_candidates
from backend.modules.breathing_detection.evaluation.evaluate_local_rate_peaks import evaluate


class LocalRatePeaksTest(unittest.TestCase):
    def test_true_low_rate_is_not_removed_by_search_boundary(self):
        t = np.arange(900) / 30
        for bpm in (7.5, 18.):
            x = np.tile(np.sin(2 * np.pi * bpm / 60 * t), (1, 4, 1))
            estimate = choose_peak(local_candidates(x))
            self.assertIsNotNone(estimate)
            self.assertLess(abs(estimate - bpm), 1.)

    def test_no_peak_can_abstain_and_prominence_filters(self):
        self.assertIsNone(choose_peak([]))
        peaks = [dict(bpm=12., height=1., prominence=.01), dict(bpm=18., height=.8, prominence=.4)]
        self.assertEqual(choose_peak(peaks), 12.)
        self.assertEqual(choose_peak(peaks, minimum_prominence=.05), 18.)

    def test_harm_counts_incorrect_corrections_as_well_as_rejections(self):
        rows = [dict(recording_id='a', baseline_error=1., reference_bpm=12., predictions={'candidate': p}) for p in (None, 20., 12.)]
        m = evaluate(rows, 'candidate')
        self.assertEqual(m['good_rejected'], 1)
        self.assertEqual(m['good_spoiled'], 1)
        self.assertAlmostEqual(m['good_harm_rate'], 2 / 3)
