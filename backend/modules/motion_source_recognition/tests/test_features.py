"""Checks on the statistics features, plus consistency checks on the generated feature files (skipped if they do not exist yet).

    backend/modules/motion_source_recognition/.venv/bin/python -m unittest discover -s backend/modules/motion_source_recognition/tests -t .
"""
import json
import unittest

import numpy as np
import pandas as pd

from backend.modules.motion_source_recognition.preprocessing.config import PREP_DIR
from backend.modules.motion_source_recognition.preprocessing.features import N_CHANNELS, STATS, column_names, statistics_features


class StatisticsFeaturesTest(unittest.TestCase):
    def test_shape_and_column_names(self):
        x = np.random.default_rng(0).normal(size=(3, N_CHANNELS, 500)).astype(np.float32)
        out = statistics_features(x)
        self.assertEqual(out.shape, (3, len(STATS) * N_CHANNELS))
        names = column_names()
        self.assertEqual(len(names), out.shape[1])
        self.assertEqual(len(set(names)), len(names))

    def test_dominant_frequency_and_band_energy_follow_a_known_sine(self):
        t = np.arange(500) / 100.0
        x = np.zeros((1, N_CHANNELS, 500), dtype=np.float32)
        x[0, 0] = 1.0 + np.sin(2 * np.pi * 5.0 * t)   # 5 Hz sits in the high band
        x[0, 1] = 1.0 + np.sin(2 * np.pi * 1.0 * t)   # 1 Hz sits in the low band
        out = statistics_features(x).reshape(len(STATS), N_CHANNELS)
        row = {name: out[i] for i, name in enumerate(STATS)}
        self.assertAlmostEqual(float(row["dominant_freq"][0]), 5.0, delta=0.21)
        self.assertAlmostEqual(float(row["dominant_freq"][1]), 1.0, delta=0.21)
        self.assertGreater(row["high_band_energy"][0], 10 * row["low_band_energy"][0])
        self.assertGreater(row["low_band_energy"][1], 10 * row["high_band_energy"][1])
        self.assertAlmostEqual(float(row["mean"][0]), 1.0, places=4)

    def test_flat_channel_gives_finite_values(self):
        x = np.ones((1, N_CHANNELS, 500), dtype=np.float32)
        self.assertTrue(np.isfinite(statistics_features(x)).all())


@unittest.skipUnless((PREP_DIR / "features_stats.npy").exists(), "run features.py first")
class GeneratedStatisticsFilesTest(unittest.TestCase):
    def test_rows_follow_the_manifest_and_are_finite(self):
        features = np.load(PREP_DIR / "features_stats.npy", mmap_mode="r")
        manifest = pd.read_csv(PREP_DIR / "manifest.csv")
        ids = pd.read_csv(PREP_DIR / "feature_ids.csv")
        self.assertEqual(features.shape[0], len(manifest))
        self.assertTrue((ids["id"] == manifest["id"]).all())
        self.assertEqual(features.shape[1], len(json.loads((PREP_DIR / "features_stats_columns.json").read_text())))
        self.assertTrue(np.isfinite(np.asarray(features[::50])).all())


@unittest.skipUnless(all((PREP_DIR / f"features_minirocket_split_{k}.done").exists() for k in range(3)), "run minirocket.py first")
class GeneratedMiniRocketFilesTest(unittest.TestCase):
    def test_each_split_has_aligned_finite_features(self):
        rows = len(pd.read_csv(PREP_DIR / "manifest.csv"))
        for k in range(3):
            features = np.load(PREP_DIR / f"features_minirocket_split_{k}.npy", mmap_mode="r")
            self.assertEqual(features.shape, (rows, 9996), f"split_{k}")
            self.assertTrue(np.isfinite(np.asarray(features[::200])).all(), f"split_{k}")
            self.assertTrue((PREP_DIR / f"minirocket_split_{k}.pkl").exists())


if __name__ == "__main__":
    unittest.main()
