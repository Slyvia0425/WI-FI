"""Checks on config.json, the one file that holds the tunable numbers of the module.

    backend/modules/motion_source_recognition/.venv/bin/python -m unittest discover -s backend/modules/motion_source_recognition/tests -t .
"""
import unittest

from backend.modules.motion_source_recognition.preprocessing import config


class ConfigTest(unittest.TestCase):
    def test_all_sections_are_read(self):
        self.assertEqual(sorted(config.SETTINGS), sorted(["_about", "models", "route_labels", "data", "splits", "features", "training", "experiments", "evaluation", "website"]))

    def test_the_window_shape_agrees_with_the_antenna_links(self):
        channels = config.SAMPLE_SHAPE[0]
        self.assertEqual(channels % config.DATA["antenna_links"], 0)
        self.assertEqual(len(config.CLASSES), len(set(config.CLASSES)))

    def test_split_fractions_leave_room_for_training(self):
        self.assertLess(config.SPLITS["test_fraction"] + config.SPLITS["val_fraction"], 0.5)
        self.assertEqual(sorted(config.SPLITS["priority"]), ["test", "train", "val"])

    def test_the_replay_steps_end_at_the_saved_model_size(self):
        self.assertEqual(config.EVALUATION["curves"]["trees_at"][-1], config.TRAINING["forest_trees"])
        self.assertEqual(config.EVALUATION["curves"]["iterations_at"][-1], config.TRAINING["max_iter"])

    def test_one_leaf_size_per_split(self):
        self.assertEqual(len(config.TRAINING["forest_minirocket_leaf_per_split"]), config.SPLITS["n_splits"])

    def test_the_unknown_budget_is_one_of_the_shares_tried(self):
        self.assertIn(config.EVALUATION["unknown_budget"], config.EVALUATION["unknown_shares"])

    def test_the_demonstration_frames_are_consistent(self):
        frames = config.WEBSITE["landing_demo"]
        self.assertEqual([f["t"] for f in frames], sorted({f["t"] for f in frames}))
        for f in frames:
            self.assertAlmostEqual(sum(f["probs"].values()), 1.0, places=6)
            self.assertEqual(f["label"], max(f["probs"], key=f["probs"].get))
            self.assertEqual(sorted(f["probs"]), sorted(config.CLASSES))

    def test_the_model_registry_matches_the_training_scripts(self):
        from backend.modules.motion_source_recognition.training import logistic_regression, random_forest, random_forest_minirocket
        expected = {logistic_regression.model_name(r) for r in ("minirocket", "statistics_physical")} | {random_forest.model_name(r) for r in ("statistics", "statistics_physical")} | {random_forest_minirocket.NAME}
        # logistic_regression.model_name("minirocket") is the main route: its folder is NAME itself
        self.assertEqual(set(config.MODELS), expected)
        for m in config.MODELS.values():
            self.assertIn(m["route"], config.ROUTE_LABELS)
            self.assertIn(m["algorithm"], ("forest", "linear"))
        self.assertEqual(list(config.FINALISTS), ["logistic_regression", "random_forest_minirocket", "random_forest"])


if __name__ == "__main__":
    unittest.main()
