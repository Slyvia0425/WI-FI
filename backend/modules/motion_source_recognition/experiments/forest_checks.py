"""Two small checks on the Random Forest, on validation rows only. Nothing is saved except a result file; the trained models are thrown away.

  trees   How many trees are enough? One forest of 600 trees per split; the validation macro-F1 when only the first n trees are averaged.
          The first 300 against the last 300 trees shows how much the result moves from chance alone.
  leaf    Why does a larger leaf lower the Human and Pet scores? Leaf size 50 and 400, each with and without the class weights.
          If the weights are the cause, the Human and Pet recall should stay high at leaf 400 once the weights are removed.

Run from the repository root:
    nice -n 15 backend/modules/motion_source_recognition/.venv/bin/python -m backend.modules.motion_source_recognition.experiments.forest_checks [--check trees|leaf] [--split K]
Writes outputs/experiments/forest_checks.json (merged with earlier results of the other check).
"""
import argparse
import json

import numpy as np

from ..preprocessing.config import CLASSES, EXPERIMENTS_DIR, EXPERIMENT_SETTINGS
from ..preprocessing.splits import N_SPLITS
from ..training.common import MODELS_DIR, Timer, load_split, score
from ..training.random_forest import SEED, TREES, make_model

OUT = EXPERIMENTS_DIR / "forest_checks.json"
TREE_COUNTS = EXPERIMENT_SETTINGS["forest_tree_counts"]


def chosen_params(k: int) -> dict:
    chosen = json.loads((MODELS_DIR / "random_forest" / f"split_{k}" / "run.json").read_text())["chosen_parameters"]
    return {key: value for key, value in chosen.items() if key != "n_estimators"}


def check_trees(k: int, threads: int) -> dict:
    data = load_split(k, "statistics")
    with Timer() as t:
        forest = make_model(chosen_params(k), data.class_weights, threads, SEED, 600).fit(data.X_train, data.y_train)
    per_tree = np.stack([tree.predict_proba(data.X_val) for tree in forest.estimators_])   # (trees, windows, classes)
    curve = {n: score(data.y_val, per_tree[:n].mean(axis=0))["macro_f1"] for n in TREE_COUNTS}
    halves = {"first_300": score(data.y_val, per_tree[:300].mean(axis=0))["macro_f1"], "last_300": score(data.y_val, per_tree[300:].mean(axis=0))["macro_f1"]}
    print(f"split_{k} trees: " + "  ".join(f"{n}:{v:.3f}" for n, v in curve.items()) + f" | halves {halves} | fit {t.seconds:.0f} s", flush=True)
    return {"macro_f1_by_tree_count": curve, "halves_of_300": halves, "fit_seconds": round(t.seconds, 1)}


def check_leaf(k: int, threads: int) -> dict:
    data = load_split(k, "statistics")
    out = {}
    for leaf in (50, 400):
        for weighted in (True, False):
            name = f"leaf{leaf}_{'weighted' if weighted else 'unweighted'}"
            params = {**chosen_params(k), "min_samples_leaf": leaf}
            with Timer() as t:
                model = make_model(params, data.class_weights if weighted else None, threads, SEED, TREES).fit(data.X_train, data.y_train)
            result = score(data.y_val, model.predict_proba(data.X_val))
            cm = np.array(result["confusion_matrix"])
            share = lambda row, cols: round(float(cm[row, cols].sum() / cm[row].sum()), 3)
            out[name] = {
                "accuracy": result["accuracy"], "macro_f1": result["macro_f1"], "fit_seconds": round(t.seconds, 1),
                "recall": {c: result["per_class"][c]["recall"] for c in CLASSES},
                "precision": {c: result["per_class"][c]["precision"] for c in CLASSES},
                "human_windows_called_irobot_or_fan": share(0, [2, 3]), "pet_windows_called_irobot_or_fan": share(1, [2, 3]),
                "human_windows_called_pet": share(0, [1]), "pet_windows_called_human": share(1, [0]),
            }
            r = out[name]
            print(f"split_{k} {name:20s} acc {r['accuracy']:.3f} F1 {r['macro_f1']:.3f} recall {r['recall']} | Human->IRobot/Fan {r['human_windows_called_irobot_or_fan']:.3f} "
                  f"Pet->IRobot/Fan {r['pet_windows_called_irobot_or_fan']:.3f} ({t.seconds:.0f} s)", flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", choices=["trees", "leaf"])
    parser.add_argument("--split", type=int, choices=range(N_SPLITS))
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for check, function in (("trees", check_trees), ("leaf", check_leaf)):
        if args.check in (None, check):
            for split in ([args.split] if args.split is not None else range(N_SPLITS)):
                results.setdefault(check, {})[f"split_{split}"] = function(split, args.threads)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1))
