"""A.2 experiment: does averaging a window's probabilities with the windows just before it help? Validation rows only; nothing is trained.

The module runs continuously, and one source (a person, a pet, a fan) usually stays for many consecutive 5-second windows. So the probabilities
of the last N windows are averaged, and the class is the largest of the averages. Only past windows are used (causal), as a running system would.

Order of windows: the `session` number in manifest.csv counts up by 1 from one window to the next inside a recording group (the large jumps of
about 1000 mark breaks). A run is a stretch of one group without a jump larger than GAP; averaging never crosses a group or a jump.
The start of a run has fewer than N windows before it and averages what exists.

Run from the repository root:
    backend/modules/motion_source_recognition/.venv/bin/python -m backend.modules.motion_source_recognition.experiments.temporal_smoothing
Writes outputs/experiments/temporal_smoothing.json and prints the tables.
"""
import json

import numpy as np
import pandas as pd

from ..calibration import reliability
from ..preprocessing.config import DATA, EXPERIMENT_SETTINGS, EXPERIMENTS_DIR, PREP_DIR
from ..preprocessing.splits import N_SPLITS
from ..training.common import CLASS_INDEX, MODELS_DIR, score
from .calibration_check import MODELS
from .model_comparison import seen_breakdown

OUT = EXPERIMENTS_DIR / "temporal_smoothing.json"
SIZES = EXPERIMENT_SETTINGS["smoothing_sizes"]   # number of windows averaged (1 = no smoothing)
GAP = DATA["run_gap"]                     # a jump in the session number larger than this ends a run


def smooth(proba: np.ndarray, group: np.ndarray, session: np.ndarray, n: int, gap: int = GAP) -> np.ndarray:
    """Mean of the probabilities of the last n windows of the same run, for every window. Rows may come in any order."""
    order = np.lexsort((session, group))
    g, s, p = group[order], session[order], proba[order].astype(np.float64)
    new_run = np.ones(len(p), dtype=bool)
    new_run[1:] = (g[1:] != g[:-1]) | (np.diff(s) > gap)
    starts = np.flatnonzero(new_run)
    ends = np.append(starts[1:], len(p))
    out = np.empty_like(p)
    for a, b in zip(starts, ends):
        c = np.vstack([np.zeros((1, p.shape[1])), np.cumsum(p[a:b], axis=0)])
        i = np.arange(b - a)
        lo = np.maximum(0, i + 1 - n)
        out[a:b] = (c[i + 1] - c[lo]) / (i + 1 - lo)[:, None]
    result = np.empty_like(out)
    result[order] = out
    return result.astype(np.float32)


if __name__ == "__main__":
    manifest, splits = pd.read_csv(PREP_DIR / "manifest.csv"), pd.read_csv(PREP_DIR / "splits.csv")
    label_of = splits.set_index("id")["label"].map(CLASS_INDEX)
    meta_all = manifest.set_index("id")
    results = {}
    for model in MODELS:
        for k in range(N_SPLITS):
            z = np.load(MODELS_DIR / model / f"split_{k}" / "probabilities.npz", allow_pickle=False)
            ids, proba = z["ids_val"], z["proba_val"]
            y, meta = label_of.loc[ids].to_numpy(), meta_all.loc[ids]
            group, session = meta["group_id"].to_numpy(), meta["session"].to_numpy()
            train_rooms = set(manifest[manifest["id"].isin(splits.loc[splits[f"split_{k}"] == "train", "id"])].environment)
            new_room = ~meta["environment"].isin(train_rooms).to_numpy()
            for n in SIZES:
                p = smooth(proba, group, session, n)
                result = score(y, p)
                user = seen_breakdown(manifest, splits, k, ids, y, p)["user"]
                results.setdefault(model, {}).setdefault(f"n{n}", {})[f"split_{k}"] = {
                    "macro_f1": result["macro_f1"], "accuracy": result["accuracy"], "recall": {c: v["recall"] for c, v in result["per_class"].items()},
                    "accuracy_seen_user": user["seen"]["accuracy"] if user["seen"] else None, "accuracy_unseen_user": user["unseen"]["accuracy"] if user["unseen"] else None,
                    "accuracy_unseen_room": round(float((p.argmax(1)[new_room] == y[new_room]).mean()), 4) if new_room.any() else None,
                    "ece": reliability(p, y)["ece"]}
    OUT.write_text(json.dumps(results, indent=1))
    mean = lambda model, n, f: np.mean([f(results[model][f"n{n}"][f"split_{k}"]) for k in range(N_SPLITS)])
    for model in MODELS:
        print(f"\n{model}  (mean over the three splits)")
        print("  N   macro-F1 (per split)         accuracy  recall Human Pet IRobot Fan        ECE")
        for n in SIZES:
            per_split = " ".join(f"{results[model][f'n{n}'][f'split_{k}']['macro_f1']:.3f}" for k in range(N_SPLITS))
            recalls = " ".join(f"{mean(model, n, lambda r, c=c: r['recall'][c]):.2f}" for c in ["Human", "Pet", "IRobot", "Fan"])
            print(f"  {n:<3d} {mean(model, n, lambda r: r['macro_f1']):.3f} ({per_split})   {mean(model, n, lambda r: r['accuracy']):.3f}     {recalls}   {mean(model, n, lambda r: r['ece']):.3f}")
        print("  unseen-user accuracy by split, N=1 -> N=12:", [(results[model]['n1'][f'split_{k}']['accuracy_unseen_user'], results[model]['n12'][f'split_{k}']['accuracy_unseen_user']) for k in range(N_SPLITS)])
        print("  split_2 unseen-room accuracy, N=1 -> N=12:", results[model]['n1']['split_2']['accuracy_unseen_room'], "->", results[model]['n12']['split_2']['accuracy_unseen_room'])
