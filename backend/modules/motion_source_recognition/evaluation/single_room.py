"""A.3 extra: one fixed room as a stand-in for one home. Train and test inside the same room, on different recordings.

For E11 (the only room with all four classes): 6,342 windows; Human 2 people / 9 recordings, Pet 2 pets / 15 recordings, IRobot 1 device / 2 recordings,
Fan 1 device / 1 recording. Five folds. Each Human and Pet recording is held out once (spread over the folds, both individuals in every fold), the two IRobot recordings are
held out in folds 0 and 1, and the single Fan recording is split by time: its last windows (after a gap of 30 windows, about 2.5 minutes) are the test in fold 2, its first 70% is
always in the training rows. The people, pets and devices of a test recording are in the training rows through their other recordings: this is "the home was calibrated
with some recordings, later recordings are recognised".

Fixed in advance: Logistic Regression on the MiniRocket features, C = 0.1, standardised on the training rows, class weights from the training rows of the fold; training windows with
the same content as a test window are dropped. The MiniRocket transformer of split 0 is used (its fitting sample holds the fewest E11 windows).

Run from the repository root:
    nice -n 15 backend/modules/motion_source_recognition/.venv/bin/python -m backend.modules.motion_source_recognition.evaluation.single_room [--room E11]
Writes outputs/evaluation/single_room_<room>.json and .md.
"""
import argparse
import json
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ..preprocessing.config import CLASSES, EVALUATION_DIR, PREP_DIR, TRAINING
from ..training.common import ROUTES
from ..training.rooms import C, TRANSFORMER_SPLIT, prepare, summarise


MAX_ITER, SEED = TRAINING["max_iter"], TRAINING["seed"]
N = len(CLASSES)


def run(room: str, threads: int) -> dict:
    features = np.load(PREP_DIR / ROUTES["minirocket"].format(k=TRANSFORMER_SPLIT), mmap_mode="r")
    _, folds = prepare(room)
    confusion_total, per_recording, fits = np.zeros((N, N), dtype=int), [], []
    for f, train, test in folds:
        counts = np.bincount(train["y"], minlength=N)
        weights = {i: float(len(train) / (N * counts[i])) for i in range(N)}
        X = np.ascontiguousarray(features[train["pos"].to_numpy()], dtype=np.float32)
        scaler = StandardScaler(copy=False).fit(X)
        with threadpool_limits(limits=threads), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = LogisticRegression(C=C, max_iter=MAX_ITER, class_weight=weights, random_state=SEED).fit(scaler.transform(X, copy=False), train["y"])
        predicted = model.predict_proba(scaler.transform(np.ascontiguousarray(features[test["pos"].to_numpy()], dtype=np.float32), copy=False)).argmax(axis=1)
        np.add.at(confusion_total, (test["y"].to_numpy(), predicted), 1)
        for gid, idx in test.reset_index(drop=True).groupby("group_id").indices.items():
            r = test.iloc[idx]
            per_recording.append({"fold": f, "class": r["label"].iloc[0], "individual": r["user"].iloc[0], "recording": r["activity"].iloc[0], "windows": int(len(r)),
                                  "recall": round(float((predicted[idx] == r["y"].to_numpy()).mean()), 3)})
        fits.append({"fold": f, "train_windows": int(len(train)), "test_windows": int(len(test)), "iterations": int(model.n_iter_[0])})
        print(f"fold {f}: train {len(train)}, test {len(test)}, iterations {fits[-1]['iterations']}", flush=True)
    return summarise(room, confusion_total, per_recording, fits)


def markdown(r: dict) -> str:
    L = [f"# One fixed room ({r['room']}), trained and tested on different recordings of the same room\n", f"Accuracy {r['accuracy']} (most frequent class: {r['majority_class_accuracy']}).\n", "| Class | Tested windows | Recall |", "|---|---|---|"]
    L += [f"| {c} | {r['tested_windows'][c]} | {r['recall'][c]:.2f} |" for c in CLASSES]
    L += ["", "Per recording:", "", "| Class | Individual | Recording | Windows | Recall |", "|---|---|---|---|---|"]
    L += [f"| {p['class']} | {p['individual']} | {p['recording']} | {p['windows']} | {p['recall']:.2f} |" for p in sorted(r["per_recording"], key=lambda p: (p["class"], p["individual"], p["recording"]))]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--room", default="E11")
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    result = run(args.room, args.threads)
    (EVALUATION_DIR / f"single_room_{args.room}.json").write_text(json.dumps(result, indent=1))
    (EVALUATION_DIR / f"single_room_{args.room}.md").write_text(markdown(result))
    print(markdown(result))
