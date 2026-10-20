"""How the five models are built, for the training stage of the website: no new model is searched or chosen here, the saved models are re-read or refitted at their saved settings.

  Random Forests (statistics, MiniRocket, statistics + physical): the validation score of the saved forest when only its first n trees vote (n = 1 to 300). The forests are aggregations of
      trees, so this shows how the aggregate settles as trees are added.
  Logistic Regressions (MiniRocket, statistics + physical): the model is refitted from scratch at its saved C with the iteration limit set to 5, 10, 25, 50, 100, 200 and 300 (the saved
      model is the 300 fit), i.e. "what the model would be if the training had been stopped there". A fit that converges before its limit ends the curve. For every point the validation
      log-loss, macro-F1 and accuracy and the training log-loss on a fixed sample of 4,000 training windows are recorded; this shows the weights settling and whether the 300-iteration
      limit stopped the fit before it converged. The last point reproduces the saved model.

Everything is measured on the validation rows of every split. Run from the repository root (about 25 minutes, mostly the Logistic Regression on MiniRocket; `--linear-only` keeps the forest curves of the existing file):
    nice -n 15 backend/modules/motion_source_recognition/.venv/bin/python -m backend.modules.motion_source_recognition.evaluation.training_curves
Writes outputs/evaluation/training_curves.json.
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ..preprocessing.config import EVALUATION, EVALUATION_DIR, MODELS, PREP_DIR, TRAINING
from ..preprocessing.splits import N_SPLITS
from ..training.common import CLASS_INDEX, MODELS_DIR, load_split, open_features, score


FORESTS = {name: m["route"] for name, m in MODELS.items() if m["algorithm"] == "forest"}
LINEARS = {name: m["route"] for name, m in MODELS.items() if m["algorithm"] == "linear"}
TREES_AT = EVALUATION["curves"]["trees_at"]
ITERATIONS_AT = EVALUATION["curves"]["iterations_at"]
TRAIN_SAMPLE = EVALUATION["curves"]["train_sample"]
MAX_ITER = TRAINING["max_iter"]


def running_mean(parts: list, at: list) -> dict:
    """Mean of the first n arrays for every n in `at`."""
    total, out = 0.0, {}
    for i, p in enumerate(parts, 1):
        total = total + p
        if i in at:
            out[i] = total / i
    return out


def validation(route: str, k: int):
    splits = pd.read_csv(PREP_DIR / "splits.csv")
    rows = np.flatnonzero((splits[f"split_{k}"] == "val").to_numpy())
    y = splits["label"].map(CLASS_INDEX).to_numpy()[rows]
    return np.ascontiguousarray(open_features(route, k)[rows], dtype=np.float32), y


def forest_curve(model_dir: str, route: str, k: int) -> list:
    model = joblib.load(MODELS_DIR / model_dir / f"split_{k}" / "model.joblib")
    X, y = validation(route, k)
    cumulative = running_mean([t.predict_proba(X) for t in model.estimators_], TREES_AT)
    out = []
    for n, p in cumulative.items():
        s = score(y, p)
        out.append({"trees": n, "macro_f1": s["macro_f1"], "accuracy": s["accuracy"]})
    return out


def linear_curve(model_dir: str, route: str, k: int) -> dict:
    run = json.loads((MODELS_DIR / model_dir / f"split_{k}" / "run.json").read_text())
    C = run["chosen_parameters"]["C"]
    data = load_split(k, route)
    scaler = StandardScaler(copy=False).fit(data.X_train)
    X_train, X_val = scaler.transform(data.X_train, copy=False), scaler.transform(data.X_val, copy=False)
    sample = np.random.default_rng(42).choice(len(X_train), min(TRAIN_SAMPLE, len(X_train)), replace=False)
    points, converged = [], False
    with threadpool_limits(limits=3), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for limit in ITERATIONS_AT:
            model = LogisticRegression(C=C, max_iter=limit, class_weight=data.class_weights, random_state=42).fit(X_train, data.y_train)
            used = int(model.n_iter_[0])
            p_val = model.predict_proba(X_val)
            s = score(data.y_val, p_val)
            points.append({"iterations": used, "limit": limit, "val_log_loss": round(float(log_loss(data.y_val, p_val, labels=[0, 1, 2, 3])), 4), "macro_f1": s["macro_f1"], "accuracy": s["accuracy"],
                           "train_log_loss": round(float(log_loss(data.y_train[sample], model.predict_proba(X_train[sample]), labels=[0, 1, 2, 3])), 4)})
            if used < limit:   # the optimiser stopped by itself: converged, later limits would give the same model
                converged = True
                break
    return {"C": C, "points": points, "converged": converged, "saved_validation_macro_f1": run["validation"]["macro_f1"], "refit_validation_macro_f1": points[-1]["macro_f1"]}


def run(linear_only: bool = False) -> dict:
    result = {"forests": {}, "linears": {}}
    if linear_only:   # keep the forest curves of the file written before
        result["forests"] = json.loads((EVALUATION_DIR / "training_curves.json").read_text())["forests"]
    for model_dir, route in ({} if linear_only else FORESTS).items():
        result["forests"][model_dir] = {f"split_{k}": forest_curve(model_dir, route, k) for k in range(N_SPLITS)}
        print(f"{model_dir}: forest curves done", flush=True)
    for model_dir, route in LINEARS.items():
        result["linears"][model_dir] = {}
        for k in range(N_SPLITS):
            result["linears"][model_dir][f"split_{k}"] = linear_curve(model_dir, route, k)
            c = result["linears"][model_dir][f"split_{k}"]
            print(f"{model_dir} split_{k}: C {c['C']}, {c['points'][-1]['iterations']} iterations, {'converged' if c['converged'] else 'limit'}, macro-F1 {c['refit_validation_macro_f1']} (saved {c['saved_validation_macro_f1']})", flush=True)
    return result


if __name__ == "__main__":
    import sys
    out = run(linear_only="--linear-only" in sys.argv)
    (EVALUATION_DIR / "training_curves.json").write_text(json.dumps(out, indent=1))
