"""Checks on the training helpers and the feature variants of the level-feature experiment.

    backend/modules/motion_source_recognition/.venv/bin/python -m unittest discover -s backend/modules/motion_source_recognition/tests -t .
"""
import unittest

import numpy as np

from backend.modules.motion_source_recognition.experiments.level_features import VARIANTS, build_variant
from backend.modules.motion_source_recognition.preprocessing.config import CLASSES, PREP_DIR
from backend.modules.motion_source_recognition.preprocessing.features import N_CHANNELS, STATS
from backend.modules.motion_source_recognition.training.common import MODELS_DIR, load_split, parse_extra, score, selection_key
from backend.modules.motion_source_recognition.training.summary import collect, table


class ScoreTest(unittest.TestCase):
    def test_perfect_and_wrong_predictions(self):
        y = np.array([0, 1, 2, 3, 0, 1])
        perfect = np.eye(4)[y]
        self.assertEqual(score(y, perfect)["macro_f1"], 1.0)
        self.assertEqual(score(y, perfect)["accuracy"], 1.0)
        always_human = np.eye(4)[np.zeros(6, dtype=int)]
        result = score(y, always_human)
        self.assertEqual(result["per_class"]["Human"]["recall"], 1.0)
        self.assertEqual(result["per_class"]["Fan"]["recall"], 0.0)
        self.assertEqual(result["confusion_matrix"][3], [1, 0, 0, 0])  # the Fan row: predicted as Human

    def test_selection_prefers_macro_f1_then_the_weaker_rare_class(self):
        a = {"macro_f1": 0.5, "per_class": {"IRobot": {"recall": 0.1}, "Fan": {"recall": 0.5}}}
        b = {"macro_f1": 0.5, "per_class": {"IRobot": {"recall": 0.3}, "Fan": {"recall": 0.3}}}
        c = {"macro_f1": 0.6, "per_class": {"IRobot": {"recall": 0.0}, "Fan": {"recall": 0.0}}}
        self.assertGreater(selection_key(b), selection_key(a))
        self.assertGreater(selection_key(c), selection_key(b))


class ParseExtraTest(unittest.TestCase):
    def test_values_become_numbers(self):
        self.assertEqual(parse_extra("min_samples_leaf=100,200"), [{"min_samples_leaf": 100}, {"min_samples_leaf": 200}])
        self.assertEqual(parse_extra("C=0.1,1e-05"), [{"C": 0.1}, {"C": 1e-05}])


class VariantTest(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(0).normal(size=(5, len(STATS) * N_CHANNELS)).astype(np.float32)

    def test_variant_sizes(self):
        sizes = {v: build_variant(self.x, v).shape[1] for v in VARIANTS}
        self.assertEqual(sizes, {"all": 2320, "no_level": 1624, "relative_level": 2088, "no_level_no_energy": 1160})

    def test_relative_level_is_distance_to_the_mean(self):
        n = N_CHANNELS
        mean = self.x[:, STATS.index("mean") * n:(STATS.index("mean") + 1) * n]
        low = self.x[:, STATS.index("min") * n:(STATS.index("min") + 1) * n]
        out = build_variant(self.x, "relative_level")
        np.testing.assert_allclose(out[:, -2 * n:-n], low - mean)


@unittest.skipUnless((PREP_DIR / "features_stats.npy").exists() and (PREP_DIR / "baselines.json").exists(), "run the preprocessing steps first")
class LoadSplitTest(unittest.TestCase):
    def test_training_and_validation_rows_follow_the_split_file(self):
        data = load_split(0, "statistics")
        self.assertEqual(data.X_train.shape[1], 2320)
        self.assertEqual(data.X_train.dtype, np.float32)
        self.assertEqual(len(data.X_train), len(data.y_train))
        self.assertEqual(set(np.unique(data.y_train)), set(range(len(CLASSES))))
        self.assertEqual(set(np.unique(data.y_val)), set(range(len(CLASSES))))
        self.assertGreater(data.class_weights[3], data.class_weights[0])  # rare classes weigh more


SAVED = [MODELS_DIR / name / f"split_{k}" / "probabilities.npz" for name in ("random_forest", "logistic_regression", "random_forest_minirocket") for k in range(3)]


@unittest.skipUnless(all(f.exists() for f in SAVED), "train the saved models first")
class SavedProbabilitiesTest(unittest.TestCase):
    def test_probabilities_cover_exactly_the_validation_and_test_rows(self):
        import pandas as pd

        splits = pd.read_csv(PREP_DIR / "splits.csv")
        for f in SAVED:
            k = int(f.parent.name.split("_")[1])
            z = np.load(f, allow_pickle=False)
            self.assertEqual(list(z["classes"]), CLASSES, f)
            for subset in ("val", "test"):
                expected = splits.loc[splits[f"split_{k}"] == subset, "id"].to_numpy()
                np.testing.assert_array_equal(z[f"ids_{subset}"], expected, err_msg=str(f))
                proba = z[f"proba_{subset}"]
                self.assertEqual(proba.shape, (len(expected), len(CLASSES)))
                np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-3)

    def test_all_models_scored_the_same_validation_rows(self):
        for k in range(3):
            first = np.load(MODELS_DIR / "random_forest" / f"split_{k}" / "probabilities.npz")["ids_val"]
            for name in ("logistic_regression", "random_forest_minirocket"):
                np.testing.assert_array_equal(first, np.load(MODELS_DIR / name / f"split_{k}" / "probabilities.npz")["ids_val"], err_msg=name)

    def test_summary_lists_every_saved_model_and_split(self):
        rows = collect()
        models = {r["model"] for r in rows}
        self.assertTrue({"logistic_regression", "random_forest", "random_forest_minirocket"} <= models)
        self.assertEqual(len(rows), 3 * len(models))   # every saved model has all three splits
        self.assertTrue(all(0 < r["macro_f1"] < 1 and r["model_megabytes"] > 0 for r in rows))
        self.assertIn("| random_forest_minirocket | 2 |", table(rows))


if __name__ == "__main__":
    unittest.main()
