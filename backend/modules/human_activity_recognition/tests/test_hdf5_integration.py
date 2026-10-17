from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import h5py
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal local environments
    h5py = None

from backend.modules.human_activity_recognition.preprocessing.hdf5_reader import list_datasets, read_csi_array


@unittest.skipUnless(h5py is not None, "h5py is required for HDF5 integration")
class HDF5IntegrationTest(unittest.TestCase):
    def test_reads_configured_key_and_single_dataset_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.h5"
            values = np.arange(24, dtype=np.float32).reshape(4, 6)
            with h5py.File(path, "w") as handle:
                handle.create_dataset("CSI_amps", data=values)
            loaded = read_csi_array(path, ["CSI_amps", "amplitudes"])
            np.testing.assert_array_equal(loaded, values)
            self.assertIn("CSI_amps", list_datasets(path))

    def test_dataset_split_loads_official_metadata_and_guards_groups(self):
        from dataclasses import replace

        from backend.modules.human_activity_recognition.experiments import load_dataset_split
        from backend.modules.human_activity_recognition.service import build_policy

        from ._helpers import config_for

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = root / "data" / "Multitask" / "HumanActivityRecognition"
            sample_root = root / "data" / "Multitask" / "sub_Human_h5"
            (task / "metadata").mkdir(parents=True)
            sample_root.mkdir(parents=True)
            rows = [
                ("standing", "train", "s1", "u1", "e1", "d1"),
                ("sitting", "val", "s2", "u2", "e2", "d2"),
                ("walking", "test", "s3", "u3", "e3", "d3"),
                ("lying", "train", "s4", "u4", "e4", "d4"),
            ]
            metadata = ["path,activity,split,session_id,user_id,environment_id,device_id"]
            for index, (activity, split, session, user, environment, device) in enumerate(rows):
                relative = f"../../sub_Human_h5/{index}.h5"
                metadata.append(f"{relative},{activity},{split},{session},{user},{environment},{device}")
                with h5py.File(sample_root / f"{index}.h5", "w") as handle:
                    handle.create_dataset("CSI_amps", data=np.ones((8, 32), dtype=np.float32))
            (task / "metadata" / "sample_metadata.csv").write_text("\n".join(metadata))
            config = replace(
                config_for(root), dataset=replace(config_for(root).dataset, expected_feature_size=8, min_frames=20)
            )
            bundle = load_dataset_split(config, build_policy(config))
            self.assertEqual(len(bundle.train.windows), 2)
            self.assertEqual(len(bundle.validation.windows), 1)
            self.assertEqual(len(bundle.test.windows), 1)
