"""A.2 comparison of the two models on the validation rows of each split. Test rows are not used.

Reads the probabilities saved by training/random_forest.py and training/logistic_regression.py and compares:
  - each model, and the average of their probabilities (soft voting)
  - the reference lines of A.1 on the same validation rows
  - how each model does on validation windows whose user, or whose environment, also appears in the training rows of the split
    (the identity shortcut: a high score only where the user or room was seen means the model recognises who or where, not the motion)

Run from the repository root:
    backend/modules/motion_source_recognition/.venv/bin/python -m backend.modules.motion_source_recognition.experiments.model_comparison
Writes outputs/experiments/model_comparison.json and prints the tables.
"""
import json

import numpy as np
import pandas as pd

from ..preprocessing.config import CLASSES, EXPERIMENTS_DIR, PREP_DIR
from ..preprocessing.splits import N_SPLITS
from ..training.common import CLASS_INDEX, MODELS_DIR, score

OUT = EXPERIMENTS_DIR
MODELS = {"random_forest": "Random Forest", "logistic_regression": "Logistic Regression"}


def load_probabilities(name: str, k: int) -> tuple:
    z = np.load(MODELS_DIR / name / f"split_{k}" / "probabilities.npz", allow_pickle=False)
    return z["ids_val"], z["proba_val"]


def seen_breakdown(manifest: pd.DataFrame, splits: pd.DataFrame, k: int, ids: np.ndarray, y: np.ndarray, proba: np.ndarray) -> dict:
    """Accuracy and per-class recall on validation windows split by 'user also in training' and 'environment also in training'."""
    column = f"split_{k}"
    train_ids = set(splits.loc[splits[column] == "train", "id"])
    train = manifest[manifest["id"].isin(train_ids)]
    meta = manifest.set_index("id").loc[ids]
    out = {}
    for key, seen_values in (("user", set(train.user)), ("environment", set(train.environment))):
        seen = meta[key].isin(seen_values).to_numpy()
        predicted = proba.argmax(axis=1)
        entry = {}
        for label, mask in (("seen", seen), ("unseen", ~seen)):
            if mask.sum() == 0:
                entry[label] = None
                continue
            recall = {c: (round(float((predicted[mask & (y == i)] == i).mean()), 3) if (mask & (y == i)).any() else None) for i, c in enumerate(CLASSES)}
            entry[label] = {"windows": int(mask.sum()), "accuracy": round(float((predicted[mask] == y[mask]).mean()), 3), "recall": recall}
        out[key] = entry
    return out


def run() -> dict:
    manifest = pd.read_csv(PREP_DIR / "manifest.csv")
    splits = pd.read_csv(PREP_DIR / "splits.csv")
    baselines = json.loads((PREP_DIR / "baselines.json").read_text())
    label_of = splits.set_index("id")["label"].map(CLASS_INDEX)
    result = {}
    for k in range(N_SPLITS):
        ids, rf = load_probabilities("random_forest", k)
        ids_lr, lr = load_probabilities("logistic_regression", k)
        assert (ids == ids_lr).all(), "the two models were scored on different validation rows"
        y = label_of.loc[ids].to_numpy()
        candidates = {"random_forest": rf, "logistic_regression": lr, "soft_voting": (rf + lr) / 2}
        entry = {"validation_windows": int(len(y)), "models": {}}
        for name, proba in candidates.items():
            s = score(y, proba)
            s["seen_vs_unseen"] = seen_breakdown(manifest, splits, k, ids, y, proba)
            entry["models"][name] = s
        entry["reference_lines"] = {n: v for n, v in baselines[f"split_{k}"]["baselines"]["val"].items() if isinstance(v, dict)}
        result[f"split_{k}"] = entry

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_comparison.json").write_text(json.dumps(result, indent=1))

    names = ["random_forest", "logistic_regression", "soft_voting"]
    rows = []
    for split, e in result.items():
        for n in names:
            m = e["models"][n]
            rows.append({"split": split, "model": n, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], **{f"recall_{c}": m["per_class"][c]["recall"] for c in CLASSES}})
        for n, b in e["reference_lines"].items():
            rows.append({"split": split, "model": n, "accuracy": b["accuracy"], "macro_f1": b["macro_f1"]})
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))
    print("\nMean over the three splits (validation rows)")
    print(table.groupby("model")[["accuracy", "macro_f1"] + [f"recall_{c}" for c in CLASSES]].mean().round(3).to_string())
    print("\nValidation windows whose user is in the training rows ('seen') or not ('unseen'): accuracy / windows")
    for split, e in result.items():
        for n in names:
            u = e["models"][n]["seen_vs_unseen"]["user"]
            print(f"{split} {n:20s}", {k: (None if v is None else (v["accuracy"], v["windows"])) for k, v in u.items()})
    return result


if __name__ == "__main__":
    run()
