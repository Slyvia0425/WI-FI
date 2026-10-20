"""A.2 step 2: Random Forest on the time-series statistics features, once per split.

Run from the repository root with the module's virtual environment:
    nice -n 15 backend/modules/motion_source_recognition/.venv/bin/python -m backend.modules.motion_source_recognition.training.random_forest [--split K]

For each split: a small grid over min_samples_leaf is fitted on the training rows, scored on the validation rows, and the best one
(macro-F1, then the weaker rare-class recall) is kept. Class weights come from baselines.json. Probabilities of the validation and the test
rows are saved; the test rows are not scored here. Output: outputs/models/random_forest/split_K/ (model.joblib, probabilities.npz, run.json).
"""
import argparse
import time

from sklearn.ensemble import RandomForestClassifier

from ..preprocessing.splits import N_SPLITS
from .common import Timer, append_tuning, load_split, load_test_features, parse_extra, previous_run, save_run, score, selection_key
from ..preprocessing.config import TRAINING

ROUTE = "statistics"
NAME = "random_forest"
DESCRIPTIONS = {"statistics": "Random Forest on time-series statistics", "statistics_physical": "Random Forest on statistics and physical features", "physical": "Random Forest on physical features"}


def model_name(route: str) -> str:
    return NAME if route == ROUTE else f"{NAME}_{route}"

SEED = TRAINING["seed"]
TREES = TRAINING["forest_trees"]
GRID = [{"min_samples_leaf": leaf, "max_features": "sqrt"} for leaf in TRAINING["forest_leaf_grid"]]


def make_model(params: dict, class_weights: dict, threads: int, seed: int = SEED, trees: int = TREES) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=trees, random_state=seed, n_jobs=threads, class_weight=class_weights, **params)


def run_split(k: int, threads: int, trees: int = TREES, grid: list | None = None, seed: int = SEED, extra: list | None = None, route: str = ROUTE) -> dict:
    """With `extra`, only these configurations are fitted and compared with the run already saved; the better one is kept."""
    name = model_name(route)
    data = load_split(k, route)
    previous = previous_run(name, k) if extra else None
    best = None
    tuning = []
    with Timer() as total:
        for params in (extra if previous else (grid or GRID)):
            with Timer() as t:
                model = make_model(params, data.class_weights, threads, seed, trees).fit(data.X_train, data.y_train)
            proba = model.predict_proba(data.X_val)
            result = score(data.y_val, proba)
            tuning.append({"parameters": params, "fit_seconds": round(t.seconds, 1), "accuracy": result["accuracy"], "macro_f1": result["macro_f1"],
                           "recall": {c: v["recall"] for c, v in result["per_class"].items()}})
            print(f"split_{k} {params}: macro-F1 {result['macro_f1']:.3f}, accuracy {result['accuracy']:.3f}, fit {t.seconds:.0f} s", flush=True)
            if best is None or selection_key(result) > selection_key(best[2]):
                best = (model, proba, result, params)
            else:
                del model
        if previous and selection_key(best[2]) <= selection_key(previous["validation"]):
            append_tuning(name, k, tuning, time.time() - total.start)
            print(f"split_{k}: the saved model stays (validation macro-F1 {previous['validation']['macro_f1']:.3f})", flush=True)
            return previous["validation"]
        model, proba_val, validation, params = best
        proba_test = model.predict_proba(load_test_features(k, route))
    if previous:  # keep the earlier tuning rows next to the new ones
        tuning = previous["tuning"] + tuning
    save_run(name, k, DESCRIPTIONS[route], model, data, proba_val, proba_test, {**params, "n_estimators": trees},
             validation, tuning, total.seconds + (previous["training_seconds_total"] if previous else 0), seed)
    print(f"split_{k}: chosen {params}, validation macro-F1 {validation['macro_f1']:.3f}", flush=True)
    return validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=int, choices=range(N_SPLITS))
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--trees", type=int, default=TREES)
    parser.add_argument("--extra", help="only try extra values next to the saved run, e.g. min_samples_leaf=100,200,400")
    parser.add_argument("--route", default=ROUTE, choices=list(DESCRIPTIONS), help="feature set; the model folder gets the route in its name unless it is the default")
    args = parser.parse_args()
    for split in ([args.split] if args.split is not None else range(N_SPLITS)):
        run_split(split, args.threads, args.trees, extra=parse_extra(args.extra) if args.extra else None, route=args.route)
