"""Step 2b of preprocessing: the reference lines every model must beat, and the class weights, per split.

Run from the repository root (needs outputs/prep/manifest.csv and splits.csv):
    python3 -m backend.modules.motion_source_recognition.preprocessing.baselines

Reference lines, all learned on the training samples of a split only and scored on its validation and test samples:
- majority class: always answer the most frequent training class.
- environment lookup: answer the most frequent training class of the sample's environment. A test environment never seen in training
  falls back to the majority class; the share of such samples is reported.
- user lookup (diagnostic, not part of the A.1 notes): the same by user. Every user belongs to one class, so this shows how much a
  split lets a model recognise the person or animal instead of the motion.
Class weights are N / (number of classes * class count) over the training samples of the split.

Writes outputs/prep/baselines.json.
"""
import json

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .config import CLASSES, PREP_DIR
from .splits import SPLIT_COLUMNS


def lookup_predictions(train: pd.DataFrame, target: pd.DataFrame, key: str, fallback: str):
    table = train.groupby(key)["label"].agg(lambda s: s.value_counts().index[0])
    seen = target[key].isin(table.index)
    predicted = target[key].map(table).fillna(fallback)
    return predicted, seen


def _score(truth: pd.Series, predicted: pd.Series) -> dict:
    return {
        "accuracy": round(float(accuracy_score(truth, predicted)), 4),
        "macro_f1": round(float(f1_score(truth, predicted, labels=CLASSES, average="macro", zero_division=0)), 4),
    }


def compute(manifest: pd.DataFrame, splits: pd.DataFrame) -> dict:
    data = manifest[["id", "user", "environment"]].merge(splits[["id", "label"] + SPLIT_COLUMNS], on="id")
    result = {}
    for column in SPLIT_COLUMNS:
        train = data[data[column] == "train"]
        counts = train["label"].value_counts()
        majority = counts.index[0]
        entry = {
            "majority_class": majority,
            "train_samples_per_class": {c: int(counts.get(c, 0)) for c in CLASSES},
            "class_weights": {c: round(float(len(train) / (len(CLASSES) * counts[c])), 4) for c in CLASSES},
            "baselines": {},
        }
        for subset in ["val", "test"]:
            target = data[data[column] == subset]
            truth = target["label"]
            majority_pred = pd.Series(majority, index=target.index)
            env_pred, env_seen = lookup_predictions(train, target, "environment", majority)
            user_pred, user_seen = lookup_predictions(train, target, "user", majority)
            entry["baselines"][subset] = {
                "samples": len(target),
                "majority_class": _score(truth, majority_pred),
                "environment_lookup": {**_score(truth, env_pred), "share_of_samples_in_unseen_environment": round(float(1 - env_seen.mean()), 4)},
                "user_lookup_diagnostic": {**_score(truth, user_pred), "share_of_samples_from_unseen_user": round(float(1 - user_seen.mean()), 4)},
            }
        result[column] = entry
    return result


def run() -> dict:
    manifest = pd.read_csv(PREP_DIR / "manifest.csv")
    splits = pd.read_csv(PREP_DIR / "splits.csv")
    result = compute(manifest, splits)
    (PREP_DIR / "baselines.json").write_text(json.dumps(result, indent=1))

    rows = []
    for column, entry in result.items():
        for subset, b in entry["baselines"].items():
            for name in ["majority_class", "environment_lookup", "user_lookup_diagnostic"]:
                rows.append({"split": column, "subset": subset, "baseline": name, **b[name]})
    table = pd.DataFrame(rows)
    show = table[table.subset == "test"].pivot_table(index="baseline", columns="split", values=["accuracy", "macro_f1"])
    print("TEST set\n", show.round(3).to_string())
    summary = table.groupby(["subset", "baseline"])[["accuracy", "macro_f1"]].agg(["mean", "std"]).round(3)
    print("\nMean and std over the three splits\n", summary.to_string())
    print("\nClass weights:", {c: e["class_weights"] for c, e in result.items()})
    return result


if __name__ == "__main__":
    run()
