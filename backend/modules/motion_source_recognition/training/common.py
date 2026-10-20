"""Shared pieces for training: loading one split of one feature route, scoring on validation data, saving results.

Rules this module follows (from the A.2 work notes):
- Training uses the training rows of a split only. Choices (parameters, features) are made on the validation rows only.
- Test rows are never scored here. Their probabilities are saved, and A.3 reports the test results.
- Metadata (user, activity, environment, device) is never a model input; only the feature matrix is.
"""
import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from ..preprocessing.config import CLASSES, MODELS_DIR, PREP_DIR

ROUTES = {   # route -> feature file(s) in outputs/prep; a tuple is joined column-wise
    "statistics": "features_stats.npy",
    "minirocket": "features_minirocket_split_{k}.npy",
    "physical": "features_physical.npy",
    "statistics_physical": ("features_stats.npy", "features_physical.npy"),
}
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}


class _Joined:
    """Feature files side by side: features[rows] joins the columns of the files for these rows."""

    def __init__(self, files):
        self.files = files

    def __getitem__(self, rows):
        return np.concatenate([np.asarray(f[rows], dtype=np.float32) for f in self.files], axis=1)


def open_features(route: str, k: int):
    """The feature matrix of a route (memory-mapped; index it with rows)."""
    names = ROUTES[route]
    if isinstance(names, str):
        return np.load(PREP_DIR / names.format(k=k), mmap_mode="r")
    return _Joined([np.load(PREP_DIR / n.format(k=k), mmap_mode="r") for n in names])


@dataclass
class SplitData:
    route: str            # which feature set: "statistics", "minirocket", "physical" or "statistics_physical"
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    ids_val: np.ndarray
    ids_test: np.ndarray
    class_weights: dict   # class index -> weight, from the training rows of this split


def _rows(splits: pd.DataFrame, column: str, subset: str) -> np.ndarray:
    return np.flatnonzero((splits[column] == subset).to_numpy())


def load_split(k: int, route: str) -> SplitData:
    """Training and validation rows of split k for one feature route. Features stay float32.

    The test matrix is not returned; use load_test_features for that.
    """
    splits = pd.read_csv(PREP_DIR / "splits.csv")
    ids = pd.read_csv(PREP_DIR / "feature_ids.csv")["id"]
    if not (ids.values == splits["id"].values).all():
        raise RuntimeError("feature rows do not follow the order of splits.csv")
    features = open_features(route, k)
    column = f"split_{k}"
    train, val, test = (_rows(splits, column, s) for s in ("train", "val", "test"))

    def take(rows):
        return np.ascontiguousarray(features[rows], dtype=np.float32)

    y = splits["label"].map(CLASS_INDEX).to_numpy()
    weights = json.loads((PREP_DIR / "baselines.json").read_text())[column]["class_weights"]
    return SplitData(
        route=route, X_train=take(train), y_train=y[train], X_val=take(val), y_val=y[val],
        ids_val=splits["id"].to_numpy()[val], ids_test=splits["id"].to_numpy()[test],
        class_weights={CLASS_INDEX[c]: float(w) for c, w in weights.items()},
    )


def load_test_features(k: int, route: str) -> np.ndarray:
    """Test rows, only to save probabilities. Nothing is scored on them in this module."""
    splits = pd.read_csv(PREP_DIR / "splits.csv")
    features = open_features(route, k)
    return np.ascontiguousarray(features[_rows(splits, f"split_{k}", "test")], dtype=np.float32)


def score(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """Validation scores: accuracy, macro-F1, per-class precision, recall and F1, and the confusion matrix (rows = true class)."""
    predicted = proba.argmax(axis=1)
    labels = list(range(len(CLASSES)))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, predicted, labels=labels, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, predicted)), 4),
        "macro_f1": round(float(f1_score(y_true, predicted, labels=labels, average="macro", zero_division=0)), 4),
        "per_class": {c: {"precision": round(float(precision[i]), 4), "recall": round(float(recall[i]), 4), "f1": round(float(f1[i]), 4), "windows": int(support[i])}
                      for i, c in enumerate(CLASSES)},
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=labels).tolist(),
    }


def confusion(y: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return np.bincount(y * len(CLASSES) + predicted, minlength=len(CLASSES) ** 2).reshape(len(CLASSES), len(CLASSES))


def human_vs_rest(y: np.ndarray, predicted: np.ndarray) -> dict:
    """Human as the positive class. False alarms are windows of another class called Human; misses are Humans called something else."""
    called, is_human = predicted == CLASS_INDEX["Human"], y == CLASS_INDEX["Human"]
    tp, fp, fn = int((called & is_human).sum()), int((called & ~is_human).sum()), int((~called & is_human).sum())
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(2 * precision * recall / max(precision + recall, 1e-12), 3),
            "false_alarms": {c: {"windows": int((called & (y == i)).sum()), "share_of_that_class": round(float((called & (y == i)).sum() / max((y == i).sum(), 1)), 3)} for i, c in enumerate(CLASSES) if c != "Human"},
            "misses_called": {c: int(((predicted == i) & is_human).sum()) for i, c in enumerate(CLASSES) if c != "Human"}}


def selection_key(result: dict) -> tuple:
    """How the best configuration is chosen on validation data: macro-F1, then the weaker of the two rare classes' recall."""
    rare = min(result["per_class"]["IRobot"]["recall"], result["per_class"]["Fan"]["recall"])
    return (result["macro_f1"], rare)


def save_run(model_dir: str, k: int, name: str, model, data: SplitData, proba_val: np.ndarray, proba_test: np.ndarray,
             params: dict, validation: dict, tuning: list, seconds: float, seed: int) -> None:
    """Writes outputs/models/<model_dir>/split_k/: model, probabilities of validation and test rows, parameters, scores, timing."""
    import joblib

    out = MODELS_DIR / model_dir / f"split_{k}"
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib", compress=3)
    np.savez_compressed(out / "probabilities.npz", classes=np.array(CLASSES), ids_val=np.asarray(data.ids_val, dtype=str), proba_val=proba_val.astype(np.float32),
                        ids_test=np.asarray(data.ids_test, dtype=str), proba_test=proba_test.astype(np.float32))
    (out / "run.json").write_text(json.dumps({
        "model": name, "feature_route": data.route, "split": k, "seed": seed, "chosen_parameters": params, "validation": validation,
        "tuning": tuning, "class_weights": {CLASSES[i]: w for i, w in data.class_weights.items()},
        "train_windows": int(len(data.y_train)), "validation_windows": int(len(data.y_val)), "features": int(data.X_train.shape[1]),
        "training_seconds_total": round(seconds, 1),
    }, indent=1, default=str))


def previous_run(model_dir: str, k: int) -> dict | None:
    """The saved run.json of this model and split, if one exists."""
    path = MODELS_DIR / model_dir / f"split_{k}" / "run.json"
    return json.loads(path.read_text()) if path.exists() else None


def append_tuning(model_dir: str, k: int, tuning: list, seconds: float) -> None:
    """Adds tuning rows to an existing run.json when a new configuration did not beat the saved one."""
    path = MODELS_DIR / model_dir / f"split_{k}" / "run.json"
    run = json.loads(path.read_text())
    run["tuning"] += tuning
    run["training_seconds_total"] = round(run["training_seconds_total"] + seconds, 1)
    path.write_text(json.dumps(run, indent=1, default=str))


def parse_extra(text: str) -> list:
    """'min_samples_leaf=100,200' -> [{'min_samples_leaf': 100}, {'min_samples_leaf': 200}]"""
    key, _, values = text.partition("=")
    return [{key: (float(v) if "." in v or "e" in v.lower() else int(v))} for v in values.split(",")]


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *exc):
        self.seconds = time.time() - self.start
