import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from backend.modules.breathing_detection.preprocessing.wire_cache import (
    build_wire_breath_cache,
    load_wire_breath_cache,
    subject_holdout_folds,
)


class FakeWireDataset:
    window_points = 300

    def __init__(self):
        rng = np.random.default_rng(7)
        self.samples = []
        for index, participant in enumerate(("01", "01", "02")):
            assessable = index != 1
            self.samples.append(
                SimpleNamespace(
                    x=rng.normal(size=(1, 234, 300)).astype(np.float32),
                    respiratory_reference=rng.normal(size=300).astype(np.float32),
                    sample_id=f"sample-{index}",
                    recording_id=f"recording-{participant}",
                    assessable=assessable,
                    quality_flags=() if assessable else ("LARGE_TIMESTAMP_GAP",),
                    metadata={
                        "participant_id": participant,
                        "room": "Room1",
                        "posture": "Foetus",
                        "environment": "Quiet",
                    },
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class WireBreathCacheTest(unittest.TestCase):
    def test_build_load_and_subject_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            source = FakeWireDataset()
            cache = build_wire_breath_cache(source, directory, checkpoint_every=1)
            loaded = load_wire_breath_cache(directory, len(source))

            self.assertEqual(cache.sequences.shape, (3, 300, 232))
            self.assertEqual(loaded.respiratory_references.shape, (3, 300))
            np.testing.assert_array_equal(loaded.weights, [1, 0, 1])
            folds = subject_holdout_folds(loaded.manifest)
            self.assertEqual(folds["subject_01"]["test"], [0, 1])
            self.assertEqual(folds["subject_01"]["train"], [2])
            self.assertTrue(
                set(folds["subject_01"]["test"]).isdisjoint(
                    folds["subject_01"]["train"]
                )
            )


if __name__ == "__main__":
    unittest.main()
