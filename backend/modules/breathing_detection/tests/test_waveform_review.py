import unittest
from types import SimpleNamespace

import numpy as np

from backend.modules.breathing_detection.evaluation.review_rate_waveforms import waveform_evidence, select_control


class WaveformReviewTest(unittest.TestCase):
    def test_clean_reference_has_consistent_peak_rate(self):
        t = np.arange(900) / 30
        signal = np.sin(2 * np.pi * .3 * t)
        sample = SimpleNamespace(respiratory_reference=signal, x=np.tile(signal, (1, 4, 1)))
        evidence, _ = waveform_evidence(sample, 18., 18.)
        self.assertAlmostEqual(evidence['reference_peak_count_bpm'], 18., delta=1.)
        self.assertIn('reference_peak_count_supports_spectral_rate', evidence['tags'])
        self.assertNotIn('reference_peak_count_disagrees_by_factor_two', evidence['tags'])

    def test_same_record_control_preferred_over_other_subject_match(self):
        case = dict(sample_id='target', recording_id='record1', participant_id='01', posture='Sitting', environment='Quiet')
        same = {**case, 'sample_id': 'same'}
        other = {**case, 'recording_id': 'record2', 'sample_id': 'other'}
        control, tier = select_control(case, [other, same])
        self.assertEqual(control['sample_id'], 'same')
        self.assertEqual(tier, 0)
