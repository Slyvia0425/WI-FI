"""A.2 extension: do the spectral-shape features, or fewer signals, help? Validation rows only; test rows are not used.

Needs outputs/experiments/spectral/ from experiments/spectral_features.py. Feature sets (each fitted with Random Forest and Logistic Regression):
  stats               the 2,320 statistics features (the existing baseline, refitted here with the same settings as the others)
  spectral            the ten spectral-shape numbers for each of the 232 channels (2,320)
  stats+spectral      both (4,640)
  link_stats          the ten statistics for each of the 4 link-mean signals (40)
  link_spectral       the ten spectral numbers for each of the 4 link-mean signals (40)
  link_both           both, 4 link-mean signals (80)
  link0_both          statistics and spectral numbers of the 58 channels of one single link (1,160)

Settings: Random Forest with 150 trees; min_samples_leaf 50 for the large sets, and the best of 5, 20, 50 (chosen on validation) for the sets with
40 or 80 features. Logistic Regression: standardised on training rows, class weights, C searched on validation along the usual ladder.

Run from the repository root:
    nice -n 15 backend/modules/motion_source_recognition/.venv/bin/python -m backend.modules.motion_source_recognition.experiments.spectral_evaluation [--split K] [--threads 2]
Writes outputs/experiments/spectral/spectral_evaluation.json.
"""
import argparse
import json

import numpy as np
import pandas as pd

from ..preprocessing.config import DATA, EXPERIMENTS_DIR, EXPERIMENT_SETTINGS, PREP_DIR, SAMPLE_SHAPE
from ..preprocessing.splits import N_SPLITS
from ..training.common import Timer, load_split, score, selection_key
from ..training.random_forest import SEED, make_model
from .fair_comparison import C_LADDER, lr_runner, search, standardise

SPECTRAL_DIR = EXPERIMENTS_DIR / "spectral"
RESULTS = SPECTRAL_DIR / "spectral_evaluation.json"
TREES = EXPERIMENT_SETTINGS["spectral_forest_trees"]
CHANNELS, LINKS = SAMPLE_SHAPE[0], DATA["antenna_links"]
SOURCES = {"stats": PREP_DIR / "features_stats.npy", "spectral": SPECTRAL_DIR / "spectral_channels.npy",
           "link_stats": SPECTRAL_DIR / "link_stats.npy", "link_spectral": SPECTRAL_DIR / "link_spectral.npy"}
VARIANTS = {   # name -> (sources, take only link 0 of the per-channel sources)
    "stats": (["stats"], False), "spectral": (["spectral"], False), "stats+spectral": (["stats", "spectral"], False),
    "link_stats": (["link_stats"], False), "link_spectral": (["link_spectral"], False), "link_both": (["link_stats", "link_spectral"], False),
    "link0_both": (["stats", "spectral"], True),
}


def link_slice(x: np.ndarray, link: int) -> np.ndarray:
    """The 58 channels of one link in every one of the 10 blocks of a per-channel matrix (statistic-major, 232 channels per block)."""
    width = CHANNELS // LINKS
    return np.concatenate([x[:, b * CHANNELS + link * width:b * CHANNELS + (link + 1) * width] for b in range(x.shape[1] // CHANNELS)], axis=1)


def matrices(variant: str, rows: np.ndarray) -> np.ndarray:
    names, one_link = VARIANTS[variant]
    parts = []
    for name in names:
        part = np.ascontiguousarray(np.load(SOURCES[name], mmap_mode="r")[rows], dtype=np.float32)
        parts.append(link_slice(part, 0) if one_link else part)
    return np.concatenate(parts, axis=1)


def evaluate(k: int, threads: int) -> dict:
    splits = pd.read_csv(PREP_DIR / "splits.csv")
    train, val = (np.flatnonzero((splits[f"split_{k}"] == s).to_numpy()) for s in ("train", "val"))
    data = load_split(k, "statistics")   # labels, class weights and validation ids; its feature matrix is not used
    assert (data.ids_val == splits["id"].to_numpy()[val]).all() and len(data.y_train) == len(train)
    entry = {}
    for variant in VARIANTS:
        X_train, X_val = matrices(variant, train), matrices(variant, val)
        small = X_train.shape[1] <= 80
        best = None
        for leaf in ((5, 20, 50) if small else (50,)):
            with Timer() as t:
                model = make_model({"min_samples_leaf": leaf, "max_features": "sqrt"}, data.class_weights, threads, SEED, TREES).fit(X_train, data.y_train)
            result = score(data.y_val, model.predict_proba(X_val))
            if best is None or selection_key(result) > selection_key(best[0]):
                best = (result, leaf, t.seconds)
        result, leaf, seconds = best
        entry[f"{variant}|random_forest"] = {"features": int(X_train.shape[1]), "parameters": {"min_samples_leaf": leaf}, "macro_f1": result["macro_f1"], "accuracy": result["accuracy"],
                                             "recall": {c: v["recall"] for c, v in result["per_class"].items()}, "fit_seconds": round(seconds, 1)}
        print(f"split_{k} {variant:15s} RF  {X_train.shape[1]:5d} features: macro-F1 {result['macro_f1']:.3f}, accuracy {result['accuracy']:.3f}, leaf {leaf}, {seconds:.0f} s", flush=True)
        X_train, X_val = standardise(X_train, X_val)
        value, tried, proba, result = search(C_LADDER, [1e-3, 1e-2, 1e-1] if not small else [1e-2, 1e-1, 1.0], lr_runner(data, X_train, X_val, threads))
        entry[f"{variant}|logistic_regression"] = {"features": int(X_train.shape[1]), "parameters": {"C": value}, "macro_f1": result["macro_f1"], "accuracy": result["accuracy"],
                                                   "recall": {c: v["recall"] for c, v in result["per_class"].items()}, "fit_seconds": round(sum(r["fit_seconds"] for r in tried), 1)}
        print(f"split_{k} {variant:15s} LR  {X_train.shape[1]:5d} features: macro-F1 {result['macro_f1']:.3f}, accuracy {result['accuracy']:.3f}, C {value}", flush=True)
    return entry


def table(results: dict) -> None:
    keys = list(results[next(iter(results))])
    print("\nValidation, mean over the splits that were run (macro-F1 per split | mean macro-F1 | mean accuracy | recall Human Pet IRobot Fan)")
    for key in keys:
        rows = [results[s][key] for s in sorted(results)]
        recalls = " ".join(f"{np.mean([r['recall'][c] for r in rows]):.2f}" for c in ["Human", "Pet", "IRobot", "Fan"])
        print(f"{key:36s} {rows[0]['features']:5d} | " + " ".join(f"{r['macro_f1']:.3f}" for r in rows) + f" | {np.mean([r['macro_f1'] for r in rows]):.3f} | {np.mean([r['accuracy'] for r in rows]):.3f} | {recalls}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=int, choices=range(N_SPLITS))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    for split in ([args.split] if args.split is not None else range(N_SPLITS)):
        entry = evaluate(split, args.threads)
        results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
        results[f"split_{split}"] = entry
        RESULTS.write_text(json.dumps(results, indent=1))
    table(json.loads(RESULTS.read_text()))
