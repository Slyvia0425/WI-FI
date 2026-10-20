"""Checks on the recorded model choice (A.3 step 1).

    backend/modules/motion_source_recognition/.venv/bin/python -m unittest discover -s backend/modules/motion_source_recognition/tests -t .
"""
import json
import unittest

from backend.modules.motion_source_recognition.evaluation.model_selection import fingerprint
from backend.modules.motion_source_recognition.preprocessing.config import EVALUATION_DIR

FILE = EVALUATION_DIR / "model_selection.json"


@unittest.skipUnless(FILE.exists(), "run evaluation.model_selection first")
class ModelSelectionTest(unittest.TestCase):
    def setUp(self):
        self.sel = json.loads(FILE.read_text())

    def test_main_model_has_the_highest_validation_macro_f1(self):
        means = {m: c["macro_f1"]["mean"] for m, c in self.sel["candidates"].items()}
        self.assertEqual(self.sel["main_model"], max(means, key=means.get))
        self.assertTrue(self.sel["validation_only"])

    def test_every_candidate_has_a_fingerprint_for_each_split_and_it_still_matches(self):
        for model, c in self.sel["candidates"].items():
            self.assertEqual(sorted(c["test_probability_fingerprint"]), ["split_0", "split_1", "split_2"])
            for k in range(3):
                self.assertEqual(c["test_probability_fingerprint"][f"split_{k}"], fingerprint(model, k), f"{model} split_{k} changed after the choice was recorded")


if __name__ == "__main__":
    unittest.main()
