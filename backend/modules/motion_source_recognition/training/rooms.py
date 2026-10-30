"""Folds for a test inside one room (one fixed home): which recordings of the room are held out, and what is left to train on.

Used by evaluation/single_room.py (the A.3 one-room test) and by experiments/physical_evaluation.py, which asked the same question with other features.
"""
import numpy as np
import pandas as pd

from ..preprocessing.config import CLASSES, EVALUATION, PREP_DIR
from .common import CLASS_INDEX

SETTINGS = EVALUATION["single_room"]
C, FOLDS, TRANSFORMER_SPLIT = SETTINGS["c"], SETTINGS["folds"], SETTINGS["transformer_split"]
FAN_TRAIN_SHARE, FAN_GAP = SETTINGS["fan_train_share"], SETTINGS["fan_gap_windows"]


def assign_folds(room_rows: pd.DataFrame) -> dict:
    """group_id -> fold. Human and Pet recordings round-robin (sorted by individual and activity), IRobot recordings to folds 0 and 1, Fan kept out (time split)."""
    fold = {}
    for label, g in room_rows.groupby("label"):
        recordings = g[["group_id", "user", "activity"]].drop_duplicates().sort_values(["user", "activity"])
        for i, gid in enumerate(recordings["group_id"]):
            fold[int(gid)] = i % FOLDS if label in ("Human", "Pet") else (i % 2 if label == "IRobot" else -1)
    return fold


def prepare(room: str) -> tuple:
    """The rows of one room with their folds, and for each fold its training and test rows (DataFrames with columns id, label, y, user, activity, group_id, pos)."""
    manifest, quality = pd.read_csv(PREP_DIR / "manifest.csv"), pd.read_csv(PREP_DIR / "quality.csv").set_index("id")
    assert (pd.read_csv(PREP_DIR / "feature_ids.csv")["id"].to_numpy() == manifest["id"].to_numpy()).all()
    manifest["pos"] = np.arange(len(manifest))   # row of the window in the feature files
    rows = manifest[manifest["environment"] == room].copy()
    rows["hash"] = quality.loc[rows["id"], "content_hash"].to_numpy()
    rows["y"] = rows["label"].map(CLASS_INDEX)
    rows["fold"] = rows["group_id"].map(assign_folds(rows))
    fan = rows[rows["label"] == "Fan"].sort_values("session")
    cut = int(len(fan) * FAN_TRAIN_SHARE)
    fan_tail, fan_gap_and_tail = set(fan["id"].iloc[cut + FAN_GAP:]), set(fan["id"].iloc[cut:])
    rows.loc[rows["id"].isin(fan_tail), "fold"] = 2
    folds = []
    for f in range(FOLDS):
        test = rows[rows["fold"] == f]
        train = rows[(rows["fold"] != f) & ~rows["hash"].isin(set(test["hash"]))]
        train = train[~train["id"].isin(fan_gap_and_tail)] if f == 2 else train[~train["id"].isin(fan_tail & set(test["id"]))]
        folds.append((f, train, test))
    return rows, folds


def summarise(room: str, confusion_total: np.ndarray, per_recording: list, fits: list) -> dict:
    true = confusion_total.sum(axis=1)
    return {"room": room, "setting": {"C": C, "folds": FOLDS, "fan_train_share": FAN_TRAIN_SHARE, "fan_gap_windows": FAN_GAP}, "tested_windows": {c: int(true[i]) for i, c in enumerate(CLASSES)},
            "recall": {c: round(float(confusion_total[i, i] / true[i]), 3) for i, c in enumerate(CLASSES)}, "accuracy": round(float(np.trace(confusion_total) / confusion_total.sum()), 3),
            "majority_class_accuracy": round(float(true.max() / true.sum()), 3), "confusion_matrix": confusion_total.tolist(), "per_recording": per_recording, "folds": fits}
