import csv
import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from backend.modules.breathing_detection.preprocessing.data import (
    BreathingDataset,
    DataValidationError,
)


class BreathingDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "metadata").mkdir()
        (self.root / "splits").mkdir()
        self.rows = []

    def tearDown(self):
        self.temp.cleanup()

    def add_sample(
        self,
        sample_id,
        shape,
        *,
        label="sleep",
        deployment="LOS_FanOn",
        values=None,
    ):
        relative = Path("sub_Human") / f"{sample_id}.h5"
        path = self.root / relative
        path.parent.mkdir(exist_ok=True)
        if values is None:
            values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        with h5py.File(path, "w") as handle:
            handle.create_dataset("CSI_amps", data=values)
        self.rows.append(
            {
                "id": sample_id,
                "file_path": f"./{relative}",
                "user": "U23",
                "activity": label,
                "environment": "E09",
                "device": "EchoPlus",
                "num_sub": str(shape[0]),
                "c_freq": "5G",
                "difficulty": "easy",
                "deplyoment": deployment,
                "session": "123",
                "is_multi_subject": "False",
                "label": label,
                "date": "",
            }
        )

    def write_index(self):
        fields = list(self.rows[0])
        with (self.root / "metadata" / "sample_metadata.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.rows)
        (self.root / "splits" / "train_id.json").write_text(
            json.dumps([row["id"] for row in self.rows]), encoding="utf-8"
        )

    def test_project_pipeline_preserves_real_dimensions(self):
        for features in (112, 208, 240, 416):
            self.add_sample(f"sample-{features}", (features, 300, 1))
        self.write_index()
        dataset = BreathingDataset(self.root, "train_id")

        self.assertEqual(len(dataset), 4)
        for index, features in enumerate((112, 208, 240, 416)):
            sample = dataset[index]
            self.assertEqual(sample.x.shape, (1, features, 300))
            self.assertAlmostEqual(float(sample.x.mean()), 0.0, places=5)
            self.assertAlmostEqual(float(sample.x.std()), 1.0, places=5)
            self.assertEqual(sample.label, 1)
            self.assertTrue(sample.assessable)

    def test_official_mode_pads_and_clips_to_published_shape(self):
        self.add_sample("small", (112, 300, 1))
        self.add_sample("large", (416, 300, 1))
        self.write_index()
        dataset = BreathingDataset(
            self.root, "train_id", mode="official_reproduction"
        )

        small, large = dataset[0], dataset[1]
        self.assertEqual(small.x.shape, (1, 500, 232))
        self.assertEqual(large.x.shape, (1, 500, 232))
        self.assertTrue(np.all(small.x[:, 300:, :] == 0))
        self.assertTrue(np.all(small.x[:, :, 112:] == 0))
        self.assertTrue(np.all(large.x[:, 300:, :] == 0))

    def test_metadata_is_explicit_and_unknown_values_stay_unknown(self):
        self.add_sample("known", (208, 300, 1), deployment="NLOS_FanOff")
        self.add_sample("unknown", (208, 300, 1), deployment="Set1")
        self.write_index()
        dataset = BreathingDataset(self.root, "train_id")

        self.assertIs(dataset[0].metadata["fan_on"], False)
        self.assertIs(dataset[0].metadata["los"], False)
        self.assertIsNone(dataset[1].metadata["fan_on"])
        self.assertIsNone(dataset[1].metadata["los"])

    def test_non_finite_and_flat_samples_are_not_assessable(self):
        bad = np.ones((208, 300, 1), dtype=np.float32)
        bad[0, 0, 0] = np.nan
        self.add_sample("bad", bad.shape, values=bad)
        self.add_sample("flat", (208, 300, 1), values=np.ones((208, 300, 1)))
        self.write_index()
        dataset = BreathingDataset(self.root, "train_id")

        self.assertFalse(dataset[0].assessable)
        self.assertIn("NON_FINITE_VALUES", dataset[0].quality_flags)
        self.assertTrue(np.isfinite(dataset[0].x).all())
        self.assertFalse(dataset[1].assessable)
        self.assertIn("FLAT_SIGNAL", dataset[1].quality_flags)

    def test_unknown_layout_fails_instead_of_reshaping(self):
        self.add_sample("wrong", (208, 299, 1))
        self.write_index()
        dataset = BreathingDataset(self.root, "train_id")
        with self.assertRaises(DataValidationError):
            _ = dataset[0]


if __name__ == "__main__":
    unittest.main()
