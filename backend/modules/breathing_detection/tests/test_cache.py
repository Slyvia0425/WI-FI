import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.modules.breathing_detection.preprocessing.cache import (
    build_temporal_cache,
    load_temporal_cache,
)
from backend.modules.breathing_detection.preprocessing.temporal import (
    extract_interpolated_subcarriers,
)


@dataclass
class FakeSample:
    x: np.ndarray
    label: int
    assessable: bool = True


class FakeDataset:
    split = "train_id"

    def __init__(self):
        rng = np.random.default_rng(4)
        self.samples = [
            FakeSample(rng.normal(size=(1, 112, 300)).astype(np.float32), 0),
            FakeSample(rng.normal(size=(1, 240, 300)).astype(np.float32), 1),
            FakeSample(np.ones((1, 208, 300), dtype=np.float32), 1, False),
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class TemporalCacheTest(unittest.TestCase):
    def test_build_load_and_quality_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = build_temporal_cache(FakeDataset(), Path(directory), checkpoint_every=1)
            self.assertEqual(cache.sequences.shape, (3, 300, 232))
            self.assertEqual(cache.sequences.dtype, np.float16)
            self.assertEqual(cache.labels.tolist(), [0, 1, 1])
            self.assertEqual(cache.weights.tolist(), [1, 1, 0])
            expected = extract_interpolated_subcarriers(FakeDataset().samples[0].x)
            np.testing.assert_allclose(
                np.asarray(cache.sequences[0], dtype=np.float32),
                expected,
                atol=5e-4,
                rtol=5e-4,
            )
            loaded = load_temporal_cache(directory, "train_id", 3)
            self.assertTrue(loaded.manifest["complete"])
            self.assertTrue(np.all(loaded.sequences[2] == 0))


if __name__ == "__main__":
    unittest.main()
