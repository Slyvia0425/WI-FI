from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from backend.modules.human_activity_recognition.config import load_config
from backend.modules.human_activity_recognition.experiments.runner import SplitBundle, WindowDataset
from backend.modules.human_activity_recognition.experiments.spectral import (
    extract_spectral_features,
    fit_spectral_ensemble,
)
from backend.modules.human_activity_recognition.preprocessing import FeatureStats, SampleRecord


class SpectralEnsembleTest(unittest.TestCase):
    def test_feature_extraction_shape_and_finite_values(self):
        windows = [np.random.default_rng(3).normal(size=(8, 32)).astype(np.float32)]
        features = extract_spectral_features(windows)
        self.assertEqual(features.shape, (1, 89))
        self.assertTrue(np.isfinite(features).all())

    def test_small_ensemble_is_learnable_and_deterministic(self):
        rng = np.random.default_rng(7)
        windows: list[np.ndarray] = []
        labels: list[int] = []
        records: list[SampleRecord] = []
        splits = ("train", "train", "train", "val", "test")
        for class_id in range(5):
            for index, split in enumerate(splits):
                window = rng.normal(0.0, 0.05, size=(8, 32)).astype(np.float32)
                time = np.linspace(0.0, 1.0, 32, endpoint=False)
                window[class_id] += (class_id + 1.0) * np.sin(2.0 * np.pi * (class_id + 2) * time)
                windows.append(window)
                labels.append(class_id)
                records.append(
                    SampleRecord(
                        sample_id=f"{split}-{class_id}-{index}",
                        path=Path(f"{class_id}-{index}.h5"),
                        session_id=f"s-{class_id}-{index}",
                        user_id=f"u-{index}",
                        environment_id=f"e-{index}",
                        device_id="fixture",
                        activity=(
                            "jumping",
                            "running",
                            "seated-breathing",
                            "walking",
                            "wavinghand",
                        )[class_id],
                        split=split,
                    )
                )
        train_indices = [index for index, record in enumerate(records) if record.split == "train"]
        validation_indices = [index for index, record in enumerate(records) if record.split == "val"]
        test_indices = [index for index, record in enumerate(records) if record.split == "test"]
        bundle = SplitBundle(
            train=WindowDataset(
                [windows[index] for index in train_indices],
                [labels[index] for index in train_indices],
                [records[index] for index in train_indices],
            ),
            validation=WindowDataset(
                [windows[index] for index in validation_indices],
                [labels[index] for index in validation_indices],
                [records[index] for index in validation_indices],
            ),
            test=WindowDataset(
                [windows[index] for index in test_indices],
                [labels[index] for index in test_indices],
                [records[index] for index in test_indices],
            ),
            stats=FeatureStats.fit(windows[index] for index in train_indices),
        )
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(env={"HAR_DATA_ROOT": tmp, "HAR_OUTPUT_ROOT": tmp})
        config = replace(
            config,
            spectral=replace(
                config.spectral,
                top_features=50,
                extra_models=2,
                trees_per_model=20,
                min_samples_leaf=1,
                max_features=0.5,
                histogram_weight=0.0,
                histogram_iterations=20,
                histogram_leaf_nodes=5,
            ),
        )
        first, metrics, _ = fit_spectral_ensemble(bundle, config, seed=42)
        second, _, _ = fit_spectral_ensemble(bundle, config, seed=42)
        self.assertGreaterEqual(metrics["test"]["macro_f1"], 0.8)
        np.testing.assert_allclose(
            first.predict_proba(bundle.test.windows),
            second.predict_proba(bundle.test.windows),
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
