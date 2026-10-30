"""A.2 side experiment (report only): does gradient boosting on the statistics features make different mistakes from the existing models?

Gradient boosting grows trees one after another, each correcting the windows the earlier ones got wrong. Searched: the number of rounds
(one fit, scored after 25 to 400 rounds) and min_samples_leaf (20, 100). Validation rows only; no test rows.

Measured as in `analyse`: macro-F1, accuracy and recall per class; accuracy on windows whose user was or was not in the training rows; split_2 only,
accuracy on the 2,634 windows from a room with no training window; how often it and each existing model are both wrong; and the macro-F1 of the average
of its probabilities with each existing model's. Result: no gain over the existing models (see the README).

Run from the repository root:
    nice -n 15 backend/modules/motion_source_recognition/.venv/bin/python -m backend.modules.motion_source_recognition.experiments.complementary [--split K]
Writes outputs/experiments/complementary/ (complementary.json and validation probabilities).
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from ..preprocessing.config import EXPERIMENTS_DIR, EXPERIMENT_SETTINGS, PREP_DIR
from ..preprocessing.splits import N_SPLITS
from ..training.common import MODELS_DIR, Timer, load_split, score, selection_key
from ..training.random_forest import SEED
from .fair_comparison import OUT as FAIR_DIR
from .model_comparison import seen_breakdown

OUT = EXPERIMENTS_DIR / "complementary"
RESULTS = OUT / "complementary.json"
REFERENCES = {   # name -> where its validation probabilities are saved
    "RF + statistics": MODELS_DIR / "random_forest" / "split_{k}" / "probabilities.npz",
    "LR + MiniRocket": MODELS_DIR / "logistic_regression" / "split_{k}" / "probabilities.npz",
    "LR + statistics": FAIR_DIR / "logistic_regression_statistics_split_{k}.npz",
    "RF + MiniRocket": FAIR_DIR / "random_forest_minirocket_split_{k}.npz",
}


def reference_probabilities(k: int, ids: np.ndarray) -> dict:
    out = {}
    for name, path in REFERENCES.items():
        z = np.load(str(path).format(k=k), allow_pickle=False)
        assert (z["ids_val"] == ids).all(), f"{name}: different validation rows"
        out[name] = z["proba_val"]
    return out


def analyse(k: int, name: str, proba: np.ndarray, data, manifest: pd.DataFrame, splits: pd.DataFrame) -> dict:
    """The common measurements of a candidate on the validation rows of split k."""
    y, ids = data.y_val, data.ids_val
    result = score(y, proba)
    predicted = proba.argmax(axis=1)
    refs = reference_probabilities(k, ids)
    meta = manifest.set_index("id").loc[ids]
    train_rooms = set(manifest[manifest["id"].isin(splits.loc[splits[f"split_{k}"] == "train", "id"])].environment)
    unseen_room = ~meta["environment"].isin(train_rooms).to_numpy()
    entry = {"macro_f1": result["macro_f1"], "accuracy": result["accuracy"], "recall": {c: v["recall"] for c, v in result["per_class"].items()},
             "seen_vs_unseen_user": seen_breakdown(manifest, splits, k, ids, y, proba)["user"],
             "unseen_room": None if not unseen_room.any() else {"windows": int(unseen_room.sum()), "accuracy": round(float((predicted[unseen_room] == y[unseen_room]).mean()), 4),
                                                                  "accuracy_other_windows": round(float((predicted[~unseen_room] == y[~unseen_room]).mean()), 4)},
             "against": {}, "average_with": {}}
    for ref, p in refs.items():
        other = p.argmax(axis=1)
        right, other_right = predicted == y, other == y
        entry["against"][ref] = {"agree": round(float((predicted == other).mean()), 3), "both_wrong": round(float((~right & ~other_right).mean()), 3),
                                 "only_candidate_right": round(float((right & ~other_right).mean()), 3), "only_reference_right": round(float((~right & other_right).mean()), 3)}
        avg = score(y, (proba + p) / 2)
        entry["average_with"][ref] = {"macro_f1": avg["macro_f1"], "accuracy": avg["accuracy"]}
    three = score(y, (proba + refs["LR + MiniRocket"] + refs["RF + MiniRocket"]) / 3)
    entry["average_with"]["LR + MiniRocket and RF + MiniRocket"] = {"macro_f1": three["macro_f1"], "accuracy": three["accuracy"]}
    return entry


def save(name: str, k: int, data, proba: np.ndarray) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{name}_split_{k}.npz", ids_val=np.asarray(data.ids_val, dtype=str), proba_val=proba.astype(np.float32))


ROUNDS = EXPERIMENT_SETTINGS["boosting_rounds"]


def part_gb(k, threads, manifest, splits):
    print(f"== gb split_{k}: gradient boosting on statistics")
    data = load_split(k, "statistics")
    tried, best = [], None
    for leaf in (20, 100):
        with Timer() as t, threadpool_limits(limits=threads):
            model = HistGradientBoostingClassifier(max_iter=max(ROUNDS), learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=leaf, class_weight=data.class_weights,
                                                   early_stopping=False, random_state=SEED).fit(data.X_train, data.y_train)
            staged = {i: p for i, p in enumerate(model.staged_predict_proba(data.X_val), 1) if i in ROUNDS}
        for rounds, proba in staged.items():
            result = score(data.y_val, proba)
            tried.append({"min_samples_leaf": leaf, "rounds": rounds, "macro_f1": result["macro_f1"], "accuracy": result["accuracy"],
                          "recall": {c: v["recall"] for c, v in result["per_class"].items()}, "fit_seconds": round(t.seconds, 1)})
            if best is None or selection_key(result) > selection_key(best[2]):
                best = ((leaf, rounds), proba, result)
        print(f"   leaf {leaf}: " + "  ".join(f"{r['rounds']}:{r['macro_f1']:.3f}" for r in tried if r["min_samples_leaf"] == leaf) + f"  ({t.seconds:.0f} s)", flush=True)
        del model
    (leaf, rounds), proba, result = best
    save("gradient_boosting_statistics", k, data, proba)
    entry = {"chosen": {"min_samples_leaf": leaf, "rounds": rounds}, "tried": tried, **analyse(k, "gradient boosting", proba, data, manifest, splits)}
    print(f"split_{k} gb: chosen leaf {leaf}, {rounds} rounds, macro-F1 {entry['macro_f1']:.3f}, accuracy {entry['accuracy']:.3f}", flush=True)
    return entry


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=int, choices=range(N_SPLITS))
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    manifest, splits = pd.read_csv(PREP_DIR / "manifest.csv"), pd.read_csv(PREP_DIR / "splits.csv")
    for split in ([args.split] if args.split is not None else range(N_SPLITS)):
        entry = part_gb(split, args.threads, manifest, splits)
        results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
        results.setdefault("gb", {})[f"split_{split}"] = entry
        OUT.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(json.dumps(results, indent=1, default=str))
