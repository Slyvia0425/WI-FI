"""A.2 check: does a model's "confidence" mean what it says? Validation rows only; nothing is trained.

A.3 will reject windows whose highest class probability is below a threshold, so that probability has to be meaningful: of the windows that get
"90% confident", about 90% should be right. For each saved model and split the windows are grouped by their highest probability and, per group,
the share of windows, the mean confidence and the real accuracy are listed. ECE (expected calibration error) is the weighted average gap between
confidence and accuracy. A gap does not mean the model is bad, only that its probability cannot be used as a percentage without correction.

Run from the repository root:
    backend/modules/motion_source_recognition/.venv/bin/python -m backend.modules.motion_source_recognition.experiments.calibration_check
Writes outputs/experiments/calibration_check.json and prints the tables.
"""
import json

import numpy as np

from ..calibration import reliability
from ..preprocessing.config import EXPERIMENT_SETTINGS, EXPERIMENTS_DIR
from ..preprocessing.splits import N_SPLITS
from ..training.common import MODELS_DIR, load_split

OUT = EXPERIMENTS_DIR / "calibration_check.json"
MODELS = EXPERIMENT_SETTINGS["calibration_models"]


if __name__ == "__main__":
    # y_val of a split does not depend on the feature route, so the cheap statistics route is enough to read the labels
    results = {}
    for k in range(N_SPLITS):
        y = load_split(k, "statistics").y_val
        for model in MODELS:
            z = np.load(MODELS_DIR / model / f"split_{k}" / "probabilities.npz", allow_pickle=False)
            results.setdefault(model, {})[f"split_{k}"] = reliability(z["proba_val"], y)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1))
    for model, per_split in results.items():
        print(f"\n{model}")
        for split, r in per_split.items():
            print(f"  {split}: accuracy {r['accuracy']:.3f}, mean confidence {r['mean_confidence']:.3f}, ECE {r['ece']:.3f}")
            print("     " + "   ".join(f"{b['confidence']}: {b['share']:.0%} of windows, acc {b['accuracy']}" for b in r["bins"] if b["share"]))
