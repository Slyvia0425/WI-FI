"""Checks on the generated splits. They need the files from preprocessing (outputs/prep/), so they skip when those are missing.

    python3 -m unittest discover -s backend/modules/motion_source_recognition/tests -t .
"""
import unittest

import pandas as pd

from backend.modules.motion_source_recognition.preprocessing.config import CLASSES, PREP_DIR
from backend.modules.motion_source_recognition.preprocessing.splits import SPLIT_COLUMNS, build_splits

FILES = [PREP_DIR / name for name in ["manifest.csv", "quality.csv", "splits.csv", "baselines.json"]]
SUBSETS = ["train", "val", "test", "duplicate"]


@unittest.skipUnless(all(f.exists() for f in FILES), "run the preprocessing steps first")
class SplitsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = pd.read_csv(PREP_DIR / "manifest.csv")
        cls.splits = pd.read_csv(PREP_DIR / "splits.csv")

    def test_every_sample_has_a_subset_in_every_split(self):
        self.assertEqual(len(self.splits), len(self.manifest))
        for column in SPLIT_COLUMNS:
            self.assertTrue(self.splits[column].isin(SUBSETS).all())

    def test_every_class_is_in_every_subset(self):
        for column in SPLIT_COLUMNS:
            for subset in ["train", "val", "test"]:
                present = set(self.splits.loc[self.splits[column] == subset, "label"])
                self.assertEqual(present, set(CLASSES), f"{column}/{subset}")

    def test_no_identical_window_is_shared_between_subsets(self):
        hashes = pd.read_csv(PREP_DIR / "quality.csv")[["id", "content_hash"]]
        data = self.splits.merge(hashes, on="id")
        for column in SPLIT_COLUMNS:
            used = data[data[column].isin(["train", "val", "test"])]
            subsets_per_content = used.groupby("content_hash")[column].nunique()
            self.assertTrue((subsets_per_content == 1).all(), column)

    def test_no_recording_group_is_in_both_test_and_training(self):
        for column in SPLIT_COLUMNS:
            test = set(self.splits.loc[self.splits[column] == "test", "group_id"])
            other = set(self.splits.loc[self.splits[column].isin(["train", "val"]), "group_id"])
            self.assertFalse(test & other, column)

    def test_no_recording_group_is_in_two_subsets(self):
        for column in SPLIT_COLUMNS:
            used = self.splits[self.splits[column] != "duplicate"]
            subsets_per_group = used.groupby("group_id")[column].nunique()
            self.assertTrue((subsets_per_group == 1).all(), column)

    def test_irobot_uses_each_group_once_as_test(self):
        irobot = self.splits[self.splits.label == "IRobot"]
        test_groups = [set(irobot.loc[irobot[c] == "test", "group_id"]) for c in SPLIT_COLUMNS]
        self.assertTrue(all(len(g) == 1 for g in test_groups))
        self.assertEqual(len(set.union(*test_groups)), len(SPLIT_COLUMNS))

    def test_splits_can_be_rebuilt_identically(self):
        rebuilt = build_splits(self.manifest.merge(pd.read_csv(PREP_DIR / "quality.csv")[["id", "content_hash"]], on="id"))
        for column in SPLIT_COLUMNS:
            self.assertTrue((rebuilt[column].values == self.splits[column].values).all(), column)


if __name__ == "__main__":
    unittest.main()
