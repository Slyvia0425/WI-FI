"""A.3 step 6: leave one room out. Every room in turn is held out of the training completely and used as the test; the main model is refitted each time.

Why: in the A.1 splits the training rows hold other recordings of the same rooms, people, pets and devices. Here the room is new. Note that almost every
person, pet and device was recorded in a single room (16 of 19 Humans, all 18 Pets, both robots, all 4 fans), so a new room also means new individuals:
the two cannot be separated in this dataset. This is also the situation of a new home.

Fixed in advance, nothing is tuned on the held-out room: Logistic Regression on the MiniRocket features, C = 0.1, 300 iterations, features
standardised on the training rows, class weights from the training rows of the fold. Training rows with the same content as a window of the held-out room are
dropped. The MiniRocket transformer of a split is fitted on a sample of that split's training windows (see preprocessing/minirocket.py); for each room the split
is used whose fitting sample holds the fewest windows of that room. That is none only for E19; for the others it is 37 to 1,190 of the 3,000 fitting windows
(the largest rooms E13, E11 and E20 are the worst). The transformer only sets its thresholds from the signal, never from labels, but it has seen some unlabelled windows
of the held-out room, which can make the result slightly optimistic.

Every window is held out exactly once, so the predictions of all folds together cover the whole dataset; they are summarised per room and pooled.
This experiment does not change the frozen model choice and nothing is selected with it.

Run from the repository root:
    nice -n 15 backend/modules/motion_source_recognition/.venv/bin/python -m backend.modules.motion_source_recognition.evaluation.leave_one_room_out [--room E11] [--threads 3]
Writes outputs/evaluation/leave_one_room_out.json and .md; finished rooms are skipped on a rerun.
"""
import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ..preprocessing.config import CLASSES, EVALUATION, EVALUATION_DIR, FEATURES, PREP_DIR, TRAINING
from ..preprocessing.splits import N_SPLITS
from ..training.common import CLASS_INDEX, ROUTES, confusion, human_vs_rest
from .error_analysis import row_normalised


C = EVALUATION["leave_one_room_out"]["c"]
MAX_ITER = TRAINING["max_iter"]
SEED = TRAINING["seed"]
FIT_PER_CLASS = FEATURES["minirocket_fit_per_class"]   # as in preprocessing/minirocket.py
FILE = EVALUATION_DIR / "leave_one_room_out.json"
N = len(CLASSES)


def fit_sample_ids(manifest: pd.DataFrame, splits: pd.DataFrame, k: int) -> set:
    """The windows the MiniRocket transformer of split k was fitted on (same sampling as preprocessing/minirocket.py)."""
    train = manifest[manifest["id"].isin(splits.loc[splits[f"split_{k}"] == "train", "id"])]
    return set(pd.concat([g.sample(min(FIT_PER_CLASS, len(g)), random_state=k) for _, g in train.groupby("label")])["id"])


def choose_split(manifest: pd.DataFrame, splits: pd.DataFrame, room: str) -> tuple:
    room_ids = set(manifest.loc[manifest["environment"] == room, "id"])
    counts = [len(room_ids & fit_sample_ids(manifest, splits, k)) for k in range(N_SPLITS)]
    return int(np.argmin(counts)), counts


def run_room(room: str, manifest: pd.DataFrame, splits: pd.DataFrame, quality: pd.DataFrame, threads: int) -> dict:
    k, fit_counts = choose_split(manifest, splits, room)
    features = np.load(PREP_DIR / ROUTES["minirocket"].format(k=k), mmap_mode="r")
    ids_order = pd.read_csv(PREP_DIR / "feature_ids.csv")["id"]
    assert (ids_order.to_numpy() == manifest["id"].to_numpy()).all(), "feature rows do not follow manifest.csv"
    held = (manifest["environment"] == room).to_numpy()
    held_hashes = set(quality.loc[quality["id"].isin(manifest.loc[held, "id"]), "content_hash"])
    same_content = manifest["id"].map(quality.set_index("id")["content_hash"]).isin(held_hashes).to_numpy()
    train_rows = np.flatnonzero(~held & ~same_content)
    test_rows = np.flatnonzero(held)
    y = manifest["label"].map(CLASS_INDEX).to_numpy()
    counts = np.bincount(y[train_rows], minlength=N)
    weights = {i: float(len(train_rows) / (N * counts[i])) for i in range(N)}
    start = time.time()
    X_train = np.ascontiguousarray(features[train_rows], dtype=np.float32)
    scaler = StandardScaler(copy=False).fit(X_train)
    X_train = scaler.transform(X_train, copy=False)
    with threadpool_limits(limits=threads), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LogisticRegression(C=C, max_iter=MAX_ITER, class_weight=weights, random_state=SEED).fit(X_train, y[train_rows])
    del X_train
    proba = model.predict_proba(scaler.transform(np.ascontiguousarray(features[test_rows], dtype=np.float32), copy=False))
    predicted = proba.argmax(axis=1)
    truth = y[test_rows]
    matrix = confusion(truth, predicted)
    majority = int(counts.argmax())
    np.savez_compressed(EVALUATION_DIR / f"leave_one_room_out_{room}.npz", ids=manifest["id"].to_numpy()[test_rows].astype(str), proba=proba.astype(np.float32))
    entry = {"room": room, "windows": int(len(test_rows)), "classes_in_room": {c: int((truth == i).sum()) for i, c in enumerate(CLASSES) if (truth == i).any()},
             "training_windows": int(len(train_rows)), "dropped_same_content": int((~held & same_content).sum()), "transformer_split": k, "held_out_windows_in_transformer_fit": fit_counts,
             "iterations": int(model.n_iter_[0]), "converged": bool(model.n_iter_[0] < MAX_ITER), "fit_seconds": round(time.time() - start),
             "accuracy": round(float((predicted == truth).mean()), 4), "majority_class_accuracy": round(float((truth == majority).mean()), 4), "majority_class": CLASSES[majority],
             "recall": {c: round(float(matrix[i, i] / matrix[i].sum()), 3) for i, c in enumerate(CLASSES) if matrix[i].sum()},
             "called": {c: int(n) for c, n in zip(CLASSES, np.bincount(predicted, minlength=N))}, "confusion_matrix": matrix.tolist()}
    print(f"{room}: {entry['windows']} windows {entry['classes_in_room']}: accuracy {entry['accuracy']:.3f} (majority {entry['majority_class_accuracy']:.3f}), recall {entry['recall']}, "
          f"fit {entry['fit_seconds']} s, {'converged' if entry['converged'] else 'NOT converged'}", flush=True)
    return entry


def pooled(rooms: dict) -> dict:
    matrix = np.sum([np.array(e["confusion_matrix"]) for e in rooms.values()], axis=0)
    tp, true, pred = np.diag(matrix), matrix.sum(axis=1), matrix.sum(axis=0)
    f1 = np.where(true + pred > 0, 2 * tp / np.maximum(true + pred, 1), 0.0)
    y = np.repeat(np.arange(N * N) // N, matrix.reshape(-1))
    p = np.repeat(np.arange(N * N) % N, matrix.reshape(-1))
    return {"windows": int(matrix.sum()), "accuracy": round(float(tp.sum() / matrix.sum()), 3), "macro_f1": round(float(f1.mean()), 3),
            "recall": {c: round(float(tp[i] / true[i]), 3) for i, c in enumerate(CLASSES)}, "confusion_matrix_rows_normalised": np.round(row_normalised(matrix), 3).tolist(),
            "human_vs_rest": human_vs_rest(y, p)}


def markdown(result: dict) -> str:
    rooms = result["rooms"]
    L = ["# Leave one room out (Logistic Regression on MiniRocket, C = 0.1)\n", "| Held-out room | Windows (classes) | Accuracy | Most frequent class | Recall of the classes in the room |", "|---|---|---|---|---|"]
    for room, e in rooms.items():
        L.append(f"| {room} | {e['windows']} ({', '.join(f'{c} {n}' for c, n in e['classes_in_room'].items())}) | {e['accuracy']:.3f} | {e['majority_class_accuracy']:.3f} ({e['majority_class']}) | "
                 + ", ".join(f"{c} {v:.2f}" for c, v in e["recall"].items()) + " |")
    if len(rooms) == 10:
        p = result["pooled"]
        h = p["human_vs_rest"]
        L += ["", "## All rooms pooled (every window held out once)\n", f"Windows {p['windows']}, accuracy {p['accuracy']}, macro-F1 {p['macro_f1']}.", "",
              "| | " + " | ".join(f"{c} recall" for c in CLASSES) + " |", "|---|" + "---|" * N, "| Pooled | " + " | ".join(f"{p['recall'][c]:.2f}" for c in CLASSES) + " |", "",
              f"Is it a person? Precision {h['precision']}, recall {h['recall']}, F1 {h['f1']}."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--room")
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    manifest, splits, quality = pd.read_csv(PREP_DIR / "manifest.csv"), pd.read_csv(PREP_DIR / "splits.csv"), pd.read_csv(PREP_DIR / "quality.csv")
    result = json.loads(FILE.read_text()) if FILE.exists() else {"setting": {"C": C, "max_iter": MAX_ITER, "model": "logistic_regression on MiniRocket"}, "rooms": {}}
    for room in ([args.room] if args.room else sorted(manifest["environment"].unique())):
        if room in result["rooms"]:
            continue
        result["rooms"][room] = run_room(room, manifest, splits, quality, args.threads)
        result["rooms"] = dict(sorted(result["rooms"].items()))
        if len(result["rooms"]) == 10:
            result["pooled"] = pooled(result["rooms"])
        EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
        FILE.write_text(json.dumps(result, indent=1))
        (EVALUATION_DIR / "leave_one_room_out.md").write_text(markdown(result))
    print(markdown(result))
