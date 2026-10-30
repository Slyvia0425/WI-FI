"""A.2 phase A: a fair comparison of the two algorithms, and the level-feature check for Logistic Regression. Validation rows only.

Until now Random Forest ran on the statistics features and Logistic Regression on the MiniRocket features, so "which is better" mixed up
the algorithm and the features. This fills the empty cells of the 2 x 2 table and repeats the level-feature check for Logistic Regression:

  a1  Logistic Regression on the statistics features
  a2  Random Forest on the MiniRocket features (the transformer of the split)
  a3  Logistic Regression on four versions of the statistics features (see level_features.py), with the C chosen in a1
  a2x a2 again for leaf sizes 3, 2, 1 where the best leaf of a2 was the smallest one tried (stops at the first value that is not better)
  room  the level-feature versions of both algorithms, scored separately on validation windows from a room with no training window

The setting (C or leaf size) is searched on the validation rows. The search starts from a few values and, while the best one sits at the
edge of what was tried, tries the next value outward on a fixed ladder, so the best value ends up between two worse ones.
Only validation probabilities are saved; test rows are not touched.

Run from the repository root:
    nice -n 15 backend/modules/motion_source_recognition/.venv/bin/python -m backend.modules.motion_source_recognition.experiments.fair_comparison [--part a1|a2|a3|a2x|room|summary] [--split K]
Writes outputs/experiments/fair_comparison/ (fair_comparison.json and validation probabilities).
"""
import argparse
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ..preprocessing.config import CLASSES, EXPERIMENTS_DIR, EXPERIMENT_SETTINGS, PREP_DIR, TRAINING
from ..preprocessing.splits import N_SPLITS
from ..training.common import MODELS_DIR, Timer, load_split, score, selection_key
from ..training.random_forest import SEED, TREES, make_model
from .level_features import VARIANTS, build_variant
from .model_comparison import seen_breakdown

OUT = EXPERIMENTS_DIR / "fair_comparison"
RESULTS = OUT / "fair_comparison.json"
C_LADDER = EXPERIMENT_SETTINGS["c_ladder"]
LEAF_LADDER = EXPERIMENT_SETTINGS["leaf_ladder"]
MAX_ITER = TRAINING["max_iter"]


def search(ladder: list, start: list, fit_score, max_extend: int = 3) -> tuple:
    """Tries `start`, then walks outward along `ladder` while the best value is the smallest or the largest tried. Returns (best value, tried rows, proba)."""
    tried, best = {}, None  # value -> row
    def run(value):
        nonlocal best
        if value in tried:
            return
        proba, result, extra = fit_score(value)
        tried[value] = {"value": value, "macro_f1": result["macro_f1"], "accuracy": result["accuracy"],
                        "recall": {c: v["recall"] for c, v in result["per_class"].items()}, **extra}
        if best is None or selection_key(result) > selection_key(best[2]):
            best = (value, proba, result)
        print(f"   {value}: macro-F1 {result['macro_f1']:.3f}, accuracy {result['accuracy']:.3f}, {extra['fit_seconds']:.0f} s", flush=True)
    for value in start:
        run(value)
    for _ in range(max_extend):
        low, high = min(tried), max(tried)
        if best[0] == high and ladder.index(high) + 1 < len(ladder):
            run(ladder[ladder.index(high) + 1])
        elif best[0] == low and ladder.index(low) > 0:
            run(ladder[ladder.index(low) - 1])
        else:
            break
    return best[0], sorted(tried.values(), key=lambda r: r["value"]), best[1], best[2]


def lr_runner(data, X_train, X_val, threads):
    def fit_score(C):
        with Timer() as t, threadpool_limits(limits=threads), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = LogisticRegression(C=C, max_iter=MAX_ITER, class_weight=data.class_weights, random_state=SEED).fit(X_train, data.y_train)
        proba = model.predict_proba(X_val)
        iterations = int(model.n_iter_[0])
        return proba, score(data.y_val, proba), {"fit_seconds": round(t.seconds, 1), "iterations": iterations, "converged": iterations < MAX_ITER}
    return fit_score


def standardise(X_train, X_val):
    scaler = StandardScaler(copy=False).fit(X_train)       # training rows only
    return scaler.transform(X_train, copy=False), scaler.transform(X_val, copy=False)


def finish(name: str, k: int, data, value, tried, proba, result, manifest, splits, extra: dict | None = None) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{name}_split_{k}.npz", ids_val=np.asarray(data.ids_val, dtype=str), proba_val=proba.astype(np.float32))
    entry = {"chosen": value, "tried": tried, "validation": result,
             "seen_vs_unseen_user": seen_breakdown(manifest, splits, k, data.ids_val, data.y_val, proba)["user"], **(extra or {})}
    print(f"split_{k} {name}: chosen {value}, validation macro-F1 {result['macro_f1']:.3f}, accuracy {result['accuracy']:.3f}", flush=True)
    return entry


def part_a1(k, threads, manifest, splits):
    print(f"== a1 split_{k}: Logistic Regression on statistics")
    data = load_split(k, "statistics")
    X_train, X_val = standardise(data.X_train, data.X_val)
    value, tried, proba, result = search(C_LADDER, [1e-4, 1e-3, 1e-2, 1e-1], lr_runner(data, X_train, X_val, threads))
    return finish("logistic_regression_statistics", k, data, value, tried, proba, result, manifest, splits)


def part_a2(k, threads, manifest, splits):
    print(f"== a2 split_{k}: Random Forest on MiniRocket")
    data = load_split(k, "minirocket")
    def fit_score(leaf):
        with Timer() as t:
            model = make_model({"min_samples_leaf": leaf, "max_features": "sqrt"}, data.class_weights, threads, SEED, TREES).fit(data.X_train, data.y_train)
        proba = model.predict_proba(data.X_val)
        return proba, score(data.y_val, proba), {"fit_seconds": round(t.seconds, 1)}
    value, tried, proba, result = search(LEAF_LADDER, [20, 50, 100], fit_score)
    return finish("random_forest_minirocket", k, data, value, tried, proba, result, manifest, splits)


def part_a3(k, threads, manifest, splits):
    print(f"== a3 split_{k}: Logistic Regression, level-feature versions")
    C = json.loads(RESULTS.read_text())["a1"][f"split_{k}"]["chosen"]
    data = load_split(k, "statistics")
    out = {}
    for variant in VARIANTS:
        X_train, X_val = standardise(build_variant(data.X_train, variant).copy(), build_variant(data.X_val, variant).copy())
        proba, result, extra = lr_runner(data, X_train, X_val, threads)(C)
        out[variant] = {"features": int(X_train.shape[1]), "macro_f1": result["macro_f1"], "accuracy": result["accuracy"],
                        "recall": {c: v["recall"] for c, v in result["per_class"].items()}, **extra}
        print(f"   {variant:20s} {X_train.shape[1]:5d} features  macro-F1 {result['macro_f1']:.3f}  accuracy {result['accuracy']:.3f}  ({extra['fit_seconds']:.0f} s)", flush=True)
    return {"C": C, "variants": out}


def part_a2_extend(k, threads, manifest, splits):
    entry = json.loads(RESULTS.read_text())["a2"][f"split_{k}"]
    print(f"== a2x split_{k}: Random Forest on MiniRocket, smaller leaves (best so far {entry['chosen']})")
    data = load_split(k, "minirocket")
    best_value, best_result, tried, best_proba = entry["chosen"], entry["validation"], list(entry["tried"]), None
    for leaf in (3, 2, 1):
        if leaf >= best_value:
            continue
        with Timer() as t:
            model = make_model({"min_samples_leaf": leaf, "max_features": "sqrt"}, data.class_weights, threads, SEED, TREES).fit(data.X_train, data.y_train)
        proba = model.predict_proba(data.X_val)
        result = score(data.y_val, proba)
        del model
        tried.append({"value": leaf, "macro_f1": result["macro_f1"], "accuracy": result["accuracy"],
                      "recall": {c: v["recall"] for c, v in result["per_class"].items()}, "fit_seconds": round(t.seconds, 1)})
        print(f"   {leaf}: macro-F1 {result['macro_f1']:.3f}, accuracy {result['accuracy']:.3f}, {t.seconds:.0f} s", flush=True)
        if selection_key(result) > selection_key(best_result):
            best_value, best_result, best_proba = leaf, result, proba
        else:
            break
    tried.sort(key=lambda r: r["value"])
    if best_proba is None:
        print(f"split_{k}: the earlier choice ({best_value}) stays", flush=True)
        return {**entry, "tried": tried}
    return finish("random_forest_minirocket", k, data, best_value, tried, best_proba, best_result, manifest, splits)


def part_room(k, threads, manifest, splits):
    """Accuracy of each model and feature version on validation windows from a room with / without training windows."""
    from sklearn.metrics import f1_score
    print(f"== room split_{k}")
    column = f"split_{k}"
    data = load_split(k, "statistics")
    meta = manifest.set_index("id").loc[data.ids_val]
    train_rooms = set(manifest[manifest["id"].isin(splits.loc[splits[column] == "train", "id"])].environment)
    unseen = ~meta["environment"].isin(train_rooms).to_numpy()
    rf_params = {key: v for key, v in json.loads((MODELS_DIR / "random_forest" / f"split_{k}" / "run.json").read_text())["chosen_parameters"].items() if key != "n_estimators"}
    C = json.loads(RESULTS.read_text())["a1"][column]["chosen"]
    out = {"unseen_room_windows": int(unseen.sum()), "seen_room_windows": int((~unseen).sum()), "unseen_rooms": sorted(meta["environment"][unseen].unique().tolist()),
           "unseen_room_classes": {c: int(((data.y_val == i) & unseen).sum()) for i, c in enumerate(CLASSES) if ((data.y_val == i) & unseen).any()}, "models": {}}

    def record(name, proba, extra=None):
        predicted = proba.argmax(axis=1)
        row = {"accuracy_unseen_room": round(float((predicted[unseen] == data.y_val[unseen]).mean()), 4), "accuracy_seen_rooms": round(float((predicted[~unseen] == data.y_val[~unseen]).mean()), 4),
               "accuracy_all": round(float((predicted == data.y_val).mean()), 4), "macro_f1_all": round(float(f1_score(data.y_val, predicted, labels=[0, 1, 2, 3], average="macro", zero_division=0)), 4), **(extra or {})}
        out["models"][name] = row
        print(f"   {name:34s} unseen room {row['accuracy_unseen_room']:.3f} | seen rooms {row['accuracy_seen_rooms']:.3f} | all {row['accuracy_all']:.3f} | macro-F1 {row['macro_f1_all']:.3f}", flush=True)

    for variant in VARIANTS:                      # Random Forest on statistics versions
        with Timer() as t:
            model = make_model(rf_params, data.class_weights, threads, SEED, TREES).fit(build_variant(data.X_train, variant), data.y_train)
        record(f"RF statistics {variant}", model.predict_proba(build_variant(data.X_val, variant)), {"fit_seconds": round(t.seconds, 1)})
        del model
    for variant in VARIANTS:                      # Logistic Regression on statistics versions
        X_train, X_val = standardise(build_variant(data.X_train, variant).copy(), build_variant(data.X_val, variant).copy())
        proba, _, extra = lr_runner(data, X_train, X_val, threads)(C)
        record(f"LR statistics {variant}", proba, {"fit_seconds": extra["fit_seconds"]})
    saved = {"RF statistics (saved, all)": np.load(MODELS_DIR / "random_forest" / f"split_{k}" / "probabilities.npz")["proba_val"],
             "LR MiniRocket (saved)": np.load(MODELS_DIR / "logistic_regression" / f"split_{k}" / "probabilities.npz")["proba_val"],
             "LR statistics (a1, saved)": np.load(OUT / f"logistic_regression_statistics_split_{k}.npz")["proba_val"],
             "RF MiniRocket (a2, saved)": np.load(OUT / f"random_forest_minirocket_split_{k}.npz")["proba_val"]}
    for name, proba in saved.items():
        record(name, proba)
    return out


def summary():
    r = json.loads(RESULTS.read_text())
    saved = {m: [json.loads((MODELS_DIR / m / f"split_{k}" / "run.json").read_text())["validation"] for k in range(N_SPLITS)] for m in ("random_forest", "logistic_regression")}
    cells = {
        "Random Forest + statistics (earlier)": saved["random_forest"],
        "Logistic Regression + statistics (a1)": [r["a1"][f"split_{k}"]["validation"] for k in range(N_SPLITS)] if "a1" in r else None,
        "Random Forest + MiniRocket (a2)": [r["a2"][f"split_{k}"]["validation"] for k in range(N_SPLITS)] if "a2" in r else None,
        "Logistic Regression + MiniRocket (earlier)": saved["logistic_regression"],
    }
    print("2 x 2 table, validation rows (macro-F1 per split, mean | accuracy mean | recall Human Pet IRobot Fan, mean)")
    for name, v in cells.items():
        if v is None:
            continue
        mean = lambda f: np.mean([f(x) for x in v])
        print(f"{name:44s} " + " ".join(f"{x['macro_f1']:.3f}" for x in v) + f" | {mean(lambda x: x['macro_f1']):.3f} | {mean(lambda x: x['accuracy']):.3f} | "
              + " ".join(f"{mean(lambda x, c=c: x['per_class'][c]['recall']):.2f}" for c in CLASSES))
    if "a3" in r:
        print("\nLogistic Regression on statistics, level-feature versions (macro-F1 per split, mean)")
        for variant in VARIANTS:
            v = [r["a3"][f"split_{k}"]["variants"][variant]["macro_f1"] for k in range(N_SPLITS)]
            print(f"{variant:20s} " + " ".join(f"{x:.3f}" for x in v) + f" | {np.mean(v):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["a1", "a2", "a3", "a2x", "room", "summary"])
    parser.add_argument("--split", type=int, choices=range(N_SPLITS))
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    if args.part == "summary":
        summary()
        raise SystemExit
    manifest, splits = pd.read_csv(PREP_DIR / "manifest.csv"), pd.read_csv(PREP_DIR / "splits.csv")
    parts = {"a1": ("a1", part_a1), "a2": ("a2", part_a2), "a3": ("a3", part_a3), "a2x": ("a2", part_a2_extend), "room": ("room", part_room)}
    for part, (key, function) in parts.items():
        if args.part == part or (args.part is None and part in ("a1", "a2", "a3")):
            for split in ([args.split] if args.split is not None else range(N_SPLITS)):
                entry = function(split, args.threads, manifest, splits)
                results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}   # re-read: other parts may have written meanwhile
                results.setdefault(key, {})[f"split_{split}"] = entry
                OUT.mkdir(parents=True, exist_ok=True)
                RESULTS.write_text(json.dumps(results, indent=1, default=str))
