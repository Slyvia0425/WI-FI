"""A.2 step 5: Random Forest on the MiniRocket features, saved in full (model, validation and test probabilities, parameters, scores).

This model was compared with the two official ones in experiments/fair_comparison.py, where min_samples_leaf was searched on the validation
rows (values below). It is the most accurate model and the best one on the unseen room, so it is kept as a finalist for A.3. This script
refits it once per split with the chosen value, so that A.3 can read its test probabilities. Nothing is searched here.

Run from the repository root with the module's virtual environment:
    nice -n 15 backend/modules/motion_source_recognition/.venv/bin/python -m backend.modules.motion_source_recognition.training.random_forest_minirocket [--split K]

Output: outputs/models/random_forest_minirocket/split_K/ (model.joblib, probabilities.npz, run.json). Test rows are not scored.
"""
import argparse

from ..preprocessing.splits import N_SPLITS
from .common import Timer, load_split, load_test_features, save_run, score
from .random_forest import SEED, TREES, make_model
from ..preprocessing.config import TRAINING

ROUTE = "minirocket"
NAME = "random_forest_minirocket"
LEAF = dict(enumerate(TRAINING["forest_minirocket_leaf_per_split"]))   # min_samples_leaf per split, chosen on the validation rows in experiments/fair_comparison.py


def run_split(k: int, threads: int, trees: int = TREES) -> dict:
    data = load_split(k, ROUTE)
    params = {"min_samples_leaf": LEAF[k], "max_features": "sqrt"}
    with Timer() as t:
        model = make_model(params, data.class_weights, threads, SEED, trees).fit(data.X_train, data.y_train)
    proba_val = model.predict_proba(data.X_val)
    validation = score(data.y_val, proba_val)
    proba_test = model.predict_proba(load_test_features(k, ROUTE))
    tuning = [{"parameters": params, "fit_seconds": round(t.seconds, 1), "accuracy": validation["accuracy"], "macro_f1": validation["macro_f1"],
               "recall": {c: v["recall"] for c, v in validation["per_class"].items()}}]
    save_run(NAME, k, "Random Forest on MiniRocket features", model, data, proba_val, proba_test, {**params, "n_estimators": trees}, validation, tuning, t.seconds, SEED)
    print(f"split_{k}: {params}, validation macro-F1 {validation['macro_f1']:.3f}, accuracy {validation['accuracy']:.3f}, fit {t.seconds:.0f} s", flush=True)
    return validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=int, choices=range(N_SPLITS))
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--trees", type=int, default=TREES)
    args = parser.parse_args()
    for split in ([args.split] if args.split is not None else range(N_SPLITS)):
        run_split(split, args.threads, args.trees)
