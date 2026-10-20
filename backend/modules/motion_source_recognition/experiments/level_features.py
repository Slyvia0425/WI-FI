"""A.2 experiment: do the level features act as a room fingerprint?

Mean, minimum and maximum say how large the signal is, and signal level differs a little between rooms. A model could use that to
recognise the room instead of the motion. This script refits the Random Forest of each split, with the parameters chosen in
training/random_forest.py, on three feature variants and compares them on the validation rows only:

  all                 all 2,320 statistics features (the training result)
  no_level            without mean, minimum and maximum
  relative_level      without the mean; minimum and maximum are replaced by their distance to the mean (range and shape stay, level goes)
  no_level_no_energy  also without the two band energies, which scale with the signal level

Run from the repository root:
    nice -n 15 backend/modules/motion_source_recognition/.venv/bin/python -m backend.modules.motion_source_recognition.experiments.level_features [--split K]
Writes outputs/experiments/level_features.json and prints a table. Test rows are not used.
"""
import argparse
import json

import numpy as np

from ..preprocessing.config import EXPERIMENTS_DIR
from ..preprocessing.features import STATS
from ..preprocessing.splits import N_SPLITS
from ..training.common import MODELS_DIR, Timer, load_split, score
from ..training.random_forest import SEED, TREES, make_model

OUT = EXPERIMENTS_DIR
VARIANTS = ["all", "no_level", "relative_level", "no_level_no_energy"]


def _block(x: np.ndarray, name: str) -> np.ndarray:
    n = x.shape[1] // len(STATS)
    i = STATS.index(name)
    return x[:, i * n:(i + 1) * n]


def build_variant(x: np.ndarray, variant: str) -> np.ndarray:
    """x has the 2,320 columns of features.py (statistic-major). Returns the columns of one variant."""
    keep = lambda names: np.concatenate([_block(x, s) for s in names], axis=1)
    if variant == "all":
        return x
    if variant == "no_level":
        return keep(["std", "skew", "kurtosis", "mean_abs_diff", "dominant_freq", "low_band_energy", "high_band_energy"])
    if variant == "relative_level":
        mean = _block(x, "mean")
        rest = keep(["std", "skew", "kurtosis", "mean_abs_diff", "dominant_freq", "low_band_energy", "high_band_energy"])
        return np.concatenate([rest, _block(x, "min") - mean, _block(x, "max") - mean], axis=1)
    if variant == "no_level_no_energy":
        return keep(["std", "skew", "kurtosis", "mean_abs_diff", "dominant_freq"])
    raise ValueError(variant)


def run_split(k: int, threads: int, trees: int) -> list:
    chosen = json.loads((MODELS_DIR / "random_forest" / f"split_{k}" / "run.json").read_text())["chosen_parameters"]
    params = {key: value for key, value in chosen.items() if key != "n_estimators"}
    data = load_split(k, "statistics")
    rows = []
    for variant in VARIANTS:
        x_train, x_val = build_variant(data.X_train, variant), build_variant(data.X_val, variant)
        with Timer() as t:
            model = make_model(params, data.class_weights, threads, SEED, trees).fit(x_train, data.y_train)
        result = score(data.y_val, model.predict_proba(x_val))
        row = {"split": k, "variant": variant, "features": int(x_train.shape[1]), "fit_seconds": round(t.seconds, 1), "accuracy": result["accuracy"],
               "macro_f1": result["macro_f1"], "recall": {c: v["recall"] for c, v in result["per_class"].items()}}
        rows.append(row)
        print(f"split_{k} {variant:20s} {row['features']:5d} features  macro-F1 {row['macro_f1']:.3f}  accuracy {row['accuracy']:.3f}  recall {row['recall']}", flush=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=int, choices=range(N_SPLITS))
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--trees", type=int, default=TREES)
    args = parser.parse_args()
    results = []
    for split in ([args.split] if args.split is not None else range(N_SPLITS)):
        results += run_split(split, args.threads, args.trees)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "level_features.json").write_text(json.dumps(results, indent=1))
