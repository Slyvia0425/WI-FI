"""A.2 step 5: one table of every saved model, built from the run.json files in outputs/models/.

Run from the repository root:
    backend/modules/motion_source_recognition/.venv/bin/python -m backend.modules.motion_source_recognition.training.summary
Writes outputs/models/summary.json and outputs/models/summary.md (validation scores only; test rows are not scored in A.2).
"""
import json

import numpy as np

from ..preprocessing.config import CLASSES
from ..preprocessing.splits import N_SPLITS
from .common import MODELS_DIR

REQUIRED_FILES = ["model.joblib", "probabilities.npz", "run.json"]


def collect() -> list:
    """One row per saved model and split, in the order the model folders are listed. Raises if a file is missing."""
    rows = []
    for folder in sorted(p for p in MODELS_DIR.iterdir() if p.is_dir()):
        for k in range(N_SPLITS):
            run_dir = folder / f"split_{k}"
            missing = [f for f in REQUIRED_FILES if not (run_dir / f).exists()]
            if missing:
                raise FileNotFoundError(f"{run_dir}: missing {missing}")
            run = json.loads((run_dir / "run.json").read_text())
            val = run["validation"]
            rows.append({"model": folder.name, "description": run["model"], "feature_route": run["feature_route"], "split": k,
                         "parameters": run["chosen_parameters"], "features": run["features"], "train_windows": run["train_windows"],
                         "validation_windows": run["validation_windows"], "macro_f1": val["macro_f1"], "accuracy": val["accuracy"],
                         "recall": {c: val["per_class"][c]["recall"] for c in CLASSES}, "training_seconds": run["training_seconds_total"],
                         "model_megabytes": round((run_dir / "model.joblib").stat().st_size / 1e6, 1)})
    return rows


def spread(values: list) -> str:
    return f"{np.mean(values):.3f} ± {np.std(values, ddof=1):.3f}"


def table(rows: list) -> str:
    lines = ["| Model | Split | Parameters | Macro-F1 | Accuracy | " + " | ".join(f"{c} recall" for c in CLASSES) + " | Training seconds | Model MB |",
             "|---|---|---|---|---|" + "---|" * len(CLASSES) + "---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['split']} | {r['parameters']} | {r['macro_f1']:.3f} | {r['accuracy']:.3f} | "
                     + " | ".join(f"{r['recall'][c]:.2f}" for c in CLASSES) + f" | {r['training_seconds']:.0f} | {r['model_megabytes']} |")
    lines += ["", "Mean ± standard deviation over the splits (sample standard deviation, n = 3).", "",
              "| Model | Macro-F1 | Accuracy |", "|---|---|---|"]
    for model in dict.fromkeys(r["model"] for r in rows):
        mine = [r for r in rows if r["model"] == model]
        lines.append(f"| {model} | {spread([r['macro_f1'] for r in mine])} | {spread([r['accuracy'] for r in mine])} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    rows = collect()
    (MODELS_DIR / "summary.json").write_text(json.dumps(rows, indent=1))
    (MODELS_DIR / "summary.md").write_text(table(rows))
    print(table(rows))
