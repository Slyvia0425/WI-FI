import unittest
import numpy as np

from backend.modules.breathing_detection.evaluation.audit_peak_evidence import peak_evidence, reference_evidence, label_evidence


class PeakEvidenceTests(unittest.TestCase):
    def test_single_frequency_has_supported_peak(self):
        t = np.arange(900)/30
        x = np.tile(np.sin(2*np.pi*.25*t), (1, 8, 1))
        original = x.copy()
        evidence = peak_evidence(x)
        self.assertLess(abs(evidence['peaks'][0]['bpm']-15), .5)
        self.assertEqual(evidence['peaks'][0]['carrier_support'], 1.)
        np.testing.assert_array_equal(x, original)

    def test_support_discriminates_carrier_groups(self):
        t = np.arange(900)/30
        x = np.array([*[np.sin(2*np.pi*.2*t)]*6, *[np.sin(2*np.pi*.5*t)]*2])[None]
        peaks = peak_evidence(x)['peaks']
        low = min(peaks, key=lambda p: abs(p['bpm']-12))
        high = min(peaks, key=lambda p: abs(p['bpm']-30))
        self.assertEqual(low['carrier_support'], .75)
        self.assertEqual(high['carrier_support'], .25)

    def test_reference_labels_do_not_modify_evidence(self):
        evidence = {'peaks': [{'bpm': 12., 'height': 1., 'carrier_support': .8}]}
        self.assertEqual(label_evidence(evidence, 12., 6.)['classification'], 'reference_near_peak_present')
        self.assertTrue(label_evidence(evidence, 12., 6.)['half_frequency_relation'])
        self.assertEqual(label_evidence(evidence, 30., 12.)['classification'], 'no_reference_near_mean_peak')
        self.assertEqual(evidence['peaks'][0]['bpm'], 12.)

    def test_empty_peaks_are_explicit(self):
        result = label_evidence({'peaks': []}, 12., 24.)
        self.assertIsNone(result['closest_peak'])
        self.assertTrue(result['double_frequency_relation'])

    def test_reference_halves_and_competing_peaks_are_descriptive(self):
        t = np.arange(900)/30
        evidence = reference_evidence(np.sin(2*np.pi*.2*t))
        self.assertFalse(evidence['competing_peak_warning'])
        self.assertFalse(evidence['temporal_discrepancy_warning'])
        self.assertEqual(len(evidence['half_window_bpm']), 2)


if __name__ == '__main__':
    unittest.main()
