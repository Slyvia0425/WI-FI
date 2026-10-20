"""A.2 step 3: Logistic Regression on the MiniRocket features, once per split.

Run from the repository root with the module's virtual environment:
    nice -n 15 backend/modules/motion_source_recognition/.venv/bin/python -m backend.modules.motion_source_recognition.training.logistic_regression [--split K]

For each split: the features are standardised with the training rows only, a small grid over the regularisation strength C is fitted on the
training rows and scored on the validation rows, and the best one (macro-F1, then the weaker rare-class recall) is kept. Class weights come
from baselines.json. Probabilities of the validation and the test rows are saved; the test rows are not scored here.
Output: outputs/models/logistic_regression/split_K/ (model.joblib = scaler + classifier, probabilities.npz, run.json).
Logistic Regression gives class probabilities directly, so no extra calibration step is needed to read a confidence from it.
"""
import argparse
import time
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ..preprocessing.splits import N_SPLITS
from .common import Timer, append_tuning, load_split, load_test_features, parse_extra, previous_run, save_run, score, selection_key
from ..preprocessing.config import TRAINING

ROUTE = "minirocket"
NAME = "logistic_regression"
SEED = TRAINING["seed"]
MAX_ITER = TRAINING["max_iter"]
GRIDS = {route: [{"C": c} for c in grid] for route, grid in TRAINING["logistic_c_grids"].items()}
GRID = GRIDS["minirocket"]
DESCRIPTIONS = {"minirocket": "Logistic Regression on MiniRocket features", "statistics_physical": "Logistic Regression on statistics and physical features", "physical": "Logistic Regression on physical features"}


def model_name(route: str) -> str:
    return NAME if route == ROUTE else f"{NAME}_{route}"



def run_split(k: int, threads: int, grid: list | None = None, max_iter: int = MAX_ITER, seed: int = SEED, extra: list | None = None, route: str = ROUTE) -> dict:
    """With `extra`, only these configurations are fitted and compared with the run already saved; the better one is kept."""
    name = model_name(route)
    data = load_split(k, route)
    previous = previous_run(name, k) if extra else None
    scaler = StandardScaler(copy=False).fit(data.X_train)      # training rows only
    X_train = scaler.transform(data.X_train, copy=False)
    X_val = scaler.transform(data.X_val, copy=False)
    best, tuning = None, []
    with Timer() as total, threadpool_limits(limits=threads):
        for params in (extra if previous else (grid or GRIDS[route])):
            with Timer() as t, warnings.catch_warnings():
                warnings.simplefilter("ignore")           # convergence is recorded below instead
                model = LogisticRegression(max_iter=max_iter, class_weight=data.class_weights, random_state=seed, **params).fit(X_train, data.y_train)
            proba = model.predict_proba(X_val)
            result = score(data.y_val, proba)
            iterations = int(model.n_iter_[0])
            tuning.append({"parameters": params, "fit_seconds": round(t.seconds, 1), "iterations": iterations, "converged": iterations < max_iter,
                           "accuracy": result["accuracy"], "macro_f1": result["macro_f1"], "recall": {c: v["recall"] for c, v in result["per_class"].items()}})
            print(f"split_{k} {params}: macro-F1 {result['macro_f1']:.3f}, accuracy {result['accuracy']:.3f}, {iterations} iterations, fit {t.seconds:.0f} s", flush=True)
            if best is None or selection_key(result) > selection_key(best[2]):
                best = (model, proba, result, params)
        if previous and selection_key(best[2]) <= selection_key(previous["validation"]):
            append_tuning(name, k, tuning, time.time() - total.start)
            print(f"split_{k}: the saved model stays (validation macro-F1 {previous['validation']['macro_f1']:.3f})", flush=True)
            return previous["validation"]
        model, proba_val, validation, params = best
        proba_test = model.predict_proba(scaler.transform(load_test_features(k, route), copy=False))
    pipeline = Pipeline([("scale", scaler), ("classifier", model)])
    if previous:  # keep the earlier tuning rows next to the new ones
        tuning = previous["tuning"] + tuning
    save_run(name, k, DESCRIPTIONS[route], pipeline, data, proba_val, proba_test, {**params, "max_iter": max_iter},
             validation, tuning, total.seconds + (previous["training_seconds_total"] if previous else 0), seed)
    print(f"split_{k}: chosen {params}, validation macro-F1 {validation['macro_f1']:.3f}", flush=True)
    return validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=int, choices=range(N_SPLITS))
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--extra", help="only try extra values next to the saved run, e.g. C=0.1")
    parser.add_argument("--route", default=ROUTE, choices=list(GRIDS), help="feature set; the model folder gets the route in its name unless it is the default")
    args = parser.parse_args()
    for split in ([args.split] if args.split is not None else range(N_SPLITS)):
        run_split(split, args.threads, extra=parse_extra(args.extra) if args.extra else None, route=args.route)
