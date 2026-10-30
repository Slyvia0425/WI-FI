import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from backend.modules.breathing_detection.preprocessing.wire_breath import (
    WireBreathDataset,
)


class WireBreathDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write_segment(self, *, environment="Quiet", segment=1, large_gap=False):
        time = np.arange(1500, dtype=np.float32) / 48.8
        if large_gap:
            time[250:] += 0.5
        gt_time = np.arange(1540, dtype=np.float32) / 50.0
        carrier_scale = np.linspace(0.5, 1.5, 234, dtype=np.float32)
        csi = np.sin(2 * np.pi * 0.25 * time[:, None]) * carrier_scale[None, :]
        csi += 0.05 * np.cos(2 * np.pi * 1.1 * time[:, None])
        gt_signal = np.sin(2 * np.pi * 0.25 * gt_time).astype(np.float32)
        file_name = (
            f"Room1_50Hz_Subject01_Foetus_{environment}_Trial01_seg{segment:03d}.mat"
        )
        # Exercise the flat archive layout whose backslashes are literal filename text.
        path = self.root / f"WiRe_Breath_Release_v1.0\\data\\Room1\\{file_name}"
        savemat(
            path,
            {
                "csi_abs": csi.astype(np.float32),
                "time": time,
                "gt_signal": gt_signal,
                "gt_time": gt_time,
                "metadata": {
                    "file_name": file_name,
                    "room": "Room1",
                    "subject_id": "01",
                    "posture": "Foetus",
                    "environment": environment,
                    "session_id": np.asarray([]),
                    "trial_id": 1,
                    "segment_id": segment,
                },
            },
        )

    def test_aligns_three_windows_and_preserves_metadata(self):
        self.write_segment()
        dataset = WireBreathDataset(self.root)

        self.assertEqual(len(dataset), 3)
        sample = dataset[0]
        self.assertEqual(sample.x.shape, (1, 234, 300))
        self.assertEqual(sample.respiratory_reference.shape, (300,))
        self.assertAlmostEqual(float(sample.x.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(sample.x.std()), 1.0, places=5)
        self.assertTrue(sample.assessable)
        self.assertEqual(sample.metadata["environment"], "Quiet")
        self.assertEqual(sample.metadata["task_label"], "breathing_present")
        self.assertIn("no-session", sample.recording_id)

    def test_marks_large_timestamp_gap_and_filters_environment(self):
        self.write_segment(environment="Interference", large_gap=True)
        dataset = WireBreathDataset(self.root, environments={"Interference"})

        self.assertFalse(dataset[0].assessable)
        self.assertIn("LARGE_TIMESTAMP_GAP", dataset[0].quality_flags)
        with self.assertRaises(ValueError):
            WireBreathDataset(self.root, environments={"Quiet"})


if __name__ == "__main__":
    unittest.main()
