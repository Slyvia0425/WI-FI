"""A.3 step 1: choose the classifier on the VALIDATION rows only, and write the choice down before any test row is scored.

Rule (fixed before looking at any test result): the main metric of the work notes is macro-F1, so the candidate with the highest mean validation
macro-F1 over the three splits is the main model. The other candidates stay in the report as comparison rows. Everything else collected here
(accuracy, per-class recall, unseen room and unseen user, probability reliability, size, training time) is context for the report and for
the risks of the choice; it does not change the rule.

Also recorded: a SHA-256 fingerprint of every saved test-probability array. Step 2 checks them, so it can be shown that the models were not
retrained or replaced between this choice and the test scoring. This script never reads a test label and never scores a test row.

Run from the repository root:
    backend/modules/motion_source_recognition/.venv/bin/python -m backend.modules.motion_source_recognition.evaluation.model_selection
Writes outputs/evaluation/model_selection.json and model_selection.md.
"""
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..experiments.model_comparison import seen_breakdown
from ..preprocessing.config import CLASSES, EVALUATION, EVALUATION_DIR, EXPERIMENTS_DIR, FINALISTS, PREP_DIR
from ..preprocessing.splits import N_SPLITS
from ..training.common import CLASS_INDEX, MODELS_DIR
from ..training.summary import collect

EXPERIMENTS = EXPERIMENTS_DIR
CANDIDATES = FINALISTS   # the models the choice is made among (config.json, models with finalist)
SMOOTHING_WINDOWS = EVALUATION["smoothing_windows"]   # the optional post-processing; reported as an extra row, not as the main numbers


def fingerprint(model: str, k: int) -> str:
    z = np.load(MODELS_DIR / model / f"split_{k}" / "probabilities.npz", allow_pickle=False)
    return hashlib.sha256(np.ascontiguousarray(z["proba_test"]).tobytes() + np.ascontiguousarray(z["ids_test"]).tobytes()).hexdigest()


def mean_std(values: list) -> dict:
    return {"mean": round(float(np.mean(values)), 3), "std": round(float(np.std(values, ddof=1)), 3), "per_split": [round(float(v), 3) for v in values]}


def build() -> dict:
    manifest, splits = pd.read_csv(PREP_DIR / "manifest.csv"), pd.read_csv(PREP_DIR / "splits.csv")
    label_of = splits.set_index("id")["label"].map(CLASS_INDEX)
    rows = collect()
    calibration = json.loads((EXPERIMENTS / "calibration_check.json").read_text())
    smoothing = json.loads((EXPERIMENTS / "temporal_smoothing.json").read_text())
    table = {}
    for model, label in CANDIDATES.items():
        mine = sorted((r for r in rows if r["model"] == model), key=lambda r: r["split"])
        unseen_room, unseen_user, seen_user = None, [], []
        for k in range(N_SPLITS):
            z = np.load(MODELS_DIR / model / f"split_{k}" / "probabilities.npz", allow_pickle=False)
            ids, proba = z["ids_val"], z["proba_val"]
            y = label_of.loc[ids].to_numpy()
            breakdown = seen_breakdown(manifest, splits, k, ids, y, proba)
            if breakdown["environment"]["unseen"]:
                unseen_room = {"split": k, **breakdown["environment"]["unseen"]}
            for kind, bucket in (("unseen", unseen_user), ("seen", seen_user)):
                bucket.append(breakdown["user"][kind]["accuracy"] if breakdown["user"][kind] else None)
        table[model] = {
            "description": label,
            "parameters_per_split": [r["parameters"] for r in mine],
            "macro_f1": mean_std([r["macro_f1"] for r in mine]),
            "accuracy": mean_std([r["accuracy"] for r in mine]),
            "recall_mean": {c: round(float(np.mean([r["recall"][c] for r in mine])), 3) for c in CLASSES},
            "recall_per_split": {c: [r["recall"][c] for r in mine] for c in ("IRobot", "Fan")},
            "accuracy_seen_user_per_split": seen_user, "accuracy_unseen_user_per_split": unseen_user,
            "unseen_room_validation": unseen_room,
            "probability_ece_per_split": [calibration[model][f"split_{k}"]["ece"] for k in range(N_SPLITS)],
            "share_of_windows_with_confidence_at_least_0.99_per_split": [next((b["share"] for b in calibration[model][f"split_{k}"]["bins"] if b["confidence"] == ">=0.99"), 0.0) for k in range(N_SPLITS)],
            "macro_f1_with_smoothing_over_6_windows": mean_std([smoothing[model][f"n{SMOOTHING_WINDOWS}"][f"split_{k}"]["macro_f1"] for k in range(N_SPLITS)]),
            "model_megabytes_per_split": [r["model_megabytes"] for r in mine],
            "training_seconds_per_split": [r["training_seconds"] for r in mine],
            "test_probability_fingerprint": {f"split_{k}": fingerprint(model, k) for k in range(N_SPLITS)},
        }
    ranking = sorted(table, key=lambda m: -table[m]["macro_f1"]["mean"])
    main = ranking[0]
    return {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rule": "main model = highest mean validation macro-F1 over the three splits; other candidates are comparison rows",
        "validation_only": True,
        "main_model": main,
        "ranking_by_validation_macro_f1": ranking,
        "gap_to_second_vs_split_spread": {"gap": round(table[ranking[0]]["macro_f1"]["mean"] - table[ranking[1]]["macro_f1"]["mean"], 3), "std_of_main_model": table[main]["macro_f1"]["std"]},
        "plan_for_the_test_scoring": [
            "all three candidates are scored on the test rows once, with the settings saved here; the test fingerprints below are checked first",
            "the main numbers are without smoothing; the smoothed version (6 windows) is an extra row",
            "confusion matrices, error analysis and the confidence threshold are done for the main model only",
            "if the test ranking differs from the validation ranking, the report says so and the main model stays",
        ],
        "candidates": table,
    }


def markdown(sel: dict) -> str:
    c = sel["candidates"]
    lines = [f"# Model selection (validation rows only)\n", f"Created {sel['created']}. Rule: {sel['rule']}.\n", f"**Main model: {c[sel['main_model']]['description']}**\n",
             "| Candidate | Macro-F1 mean ± std | Accuracy mean ± std | IRobot recall | Fan recall | Unseen-room accuracy (split_2) | ECE per split | Model MB |", "|---|---|---|---|---|---|---|---|"]
    for m in sel["ranking_by_validation_macro_f1"]:
        r = c[m]
        room = r["unseen_room_validation"]
        lines.append(f"| {r['description']} | {r['macro_f1']['mean']:.3f} ± {r['macro_f1']['std']:.3f} | {r['accuracy']['mean']:.3f} ± {r['accuracy']['std']:.3f} | {r['recall_mean']['IRobot']:.2f} | {r['recall_mean']['Fan']:.2f} | "
                     f"{room['accuracy'] if room else '-'} | {r['probability_ece_per_split']} | {max(r['model_megabytes_per_split'])} |")
    g = sel["gap_to_second_vs_split_spread"]
    lines += ["", f"Gap between the first and second candidate: {g['gap']:.3f}; spread (std) of the main model over the splits: {g['std_of_main_model']:.3f}.", "", "Plan for the test scoring:"]
    lines += [f"- {p}" for p in sel["plan_for_the_test_scoring"]]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    selection = build()
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    (EVALUATION_DIR / "model_selection.json").write_text(json.dumps(selection, indent=1))
    (EVALUATION_DIR / "model_selection.md").write_text(markdown(selection))
    print(markdown(selection))
