from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from backend.modules.human_activity_recognition.scripts.prepare_local_subset import main


class PrepareLocalSubsetTest(unittest.TestCase):
    def test_archive_extracts_only_module_b_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "csi-bench.zip"
            destination = root / "subset"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("Multitask/HumanActivityRecognition/metadata/sample.csv", "sample_id\n1\n")
                handle.writestr("Multitask/sub_Human_h5/sample.h5", b"h5")
                handle.writestr("Multitask/HumanIdentification/metadata/sample.csv", "ignored")
                handle.writestr("FallDetection/metadata/sample.csv", "ignored")
            code = main(["--archive", str(archive), "--destination", str(destination)])
            self.assertEqual(code, 0)
            self.assertTrue((destination / "Multitask/HumanActivityRecognition/metadata/sample.csv").exists())
            self.assertTrue((destination / "Multitask/sub_Human_h5/sample.h5").exists())
            self.assertFalse((destination / "FallDetection").exists())
            self.assertFalse((destination / "Multitask/HumanIdentification").exists())
            manifest = json.loads((destination / "subset_manifest.json").read_text())
            self.assertEqual(manifest["file_count"], 2)
