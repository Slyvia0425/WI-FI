import unittest
from types import SimpleNamespace

import numpy as np

from backend.modules.breathing_detection.evaluation.analyze_wire_breath_failures import (
    gap_scan,
    signal_features,
)


class WireSignalAnalysisTest(unittest.TestCase):
    def test_signal_features_find_reference_frequency(self):
        time = np.arange(300) / 30.0
        signal = np.sin(2 * np.pi * 0.3 * time).astype(np.float32)
        sample = SimpleNamespace(
            x=np.tile(signal, (1, 234, 1)),
            respiratory_reference=signal,
            metadata={"sampling_rate_hz": 30.0},
        )
        result = signal_features(sample, probability=0.9, threshold=0.5)

        self.assertAlmostEqual(result["reference_bpm"], 18.0)
        self.assertEqual(result["subcarrier_modal_peak_fraction"], 1.0)
        self.assertEqual(result["subcarrier_gt_agreement_fraction"], 1.0)

    def test_gap_scan_changes_coverage_without_relabelling(self):
        samples = [{"room": "Room1"}, {"room": "Room2"}]
        rows = gap_scan(
            samples,
            np.asarray([0.9, 0.9]),
            np.asarray([0.9, 0.05]),
            threshold=0.5,
        )
        at_100_ms = next(row for row in rows if row["maximum_gap_ms"] == 100)
        at_1000_ms = next(row for row in rows if row["maximum_gap_ms"] == 1000)
        self.assertEqual(at_100_ms["assessable"], 1)
        self.assertEqual(at_1000_ms["assessable"], 2)


if __name__ == "__main__":
    unittest.main()
