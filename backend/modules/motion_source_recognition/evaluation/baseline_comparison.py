"""A.3 step 3: the main model against the two reference lines, on the TEST rows. No model is trained or scored again.

Reference lines (learned on the training rows of each split only, as in preprocessing/baselines.py):
  majority class      always answer the most frequent training class
  environment lookup  answer the most frequent training class of the window's room; a room never seen in training falls back to the majority class

The predictions of each line are rebuilt window by window with the same code as preprocessing/baselines.py and must reproduce the saved scores
exactly; otherwise the run stops. The main model's test probabilities come from the frozen choice (evaluation/model_selection.py) and its scores from
evaluation/test_scoring.py. The gap between the model and a line is given with a 95% range from resampling recording groups within each class
(1,000 resamples, the same draws for the model and both lines, so the gap is paired).

Run from the repository root:
    backend/modules/motion_source_recognition/.venv/bin/python -m backend.modules.motion_source_recognition.evaluation.baseline_comparison
Writes outputs/evaluation/baseline_comparison.json and baseline_comparison.md.
"""
import json

import numpy as np
import pandas as pd

from ..preprocessing.baselines import lookup_predictions
from ..preprocessing.config import CLASSES, EVALUATION_DIR, PREP_DIR
from ..preprocessing.splits import N_SPLITS
from ..training.common import CLASS_INDEX, MODELS_DIR, score
from .model_selection import CANDIDATES
from .test_scoring import BOOTSTRAPS, SEED, bootstrap, confusion_by_group, interval, stratified_draws

N = len(CLASSES)
LINES = ["majority_class", "environment_lookup"]
NAMES = {"majority_class": "Always the most frequent class", "environment_lookup": "Room lookup"}


def per_class(confusion: np.ndarray) -> dict:
    tp, true, pred = np.diag(confusion), confusion.sum(axis=1), confusion.sum(axis=0)
    return {c: {"recall": round(float(tp[i] / true[i]), 3) if true[i] else None, "f1": round(float(2 * tp[i] / (true[i] + pred[i])), 3) if true[i] + pred[i] else 0.0,
                "predicted_windows": int(pred[i])} for i, c in enumerate(CLASSES)}


def run() -> dict:
    selection = json.loads((EVALUATION_DIR / "model_selection.json").read_text())
    main = selection["main_model"]
    manifest, splits = pd.read_csv(PREP_DIR / "manifest.csv"), pd.read_csv(PREP_DIR / "splits.csv")
    saved = json.loads((PREP_DIR / "baselines.json").read_text())
    data = manifest[["id", "user", "environment"]].merge(splits[["id", "label", "group_id"] + [f"split_{k}" for k in range(N_SPLITS)]], on="id")
    rng = np.random.default_rng(SEED)
    result = {"main_model": main, "bootstrap": {"resamples": BOOTSTRAPS, "seed": SEED, "unit": "recording group, resampled within each class"}, "splits": {}}
    replicates = {name: [] for name in ["model"] + LINES}
    for k in range(N_SPLITS):
        column = f"split_{k}"
        train, test = data[data[column] == "train"], data[data[column] == "test"]
        majority = train["label"].value_counts().index[0]
        z = np.load(MODELS_DIR / main / column / "probabilities.npz", allow_pickle=False)
        assert (z["ids_test"] == test["id"].to_numpy()).all(), "test rows differ from splits.csv"
        env, env_seen = lookup_predictions(train, test, "environment", majority)
        predictions = {"model": z["proba_test"].argmax(axis=1),
                       "majority_class": np.full(len(test), CLASS_INDEX[majority]),
                       "environment_lookup": env.map(CLASS_INDEX).to_numpy()}
        y, group = test["label"].map(CLASS_INDEX).to_numpy(), test["group_id"].to_numpy()
        entry, per_group = {"majority_class_name": majority, "share_of_windows_in_unseen_room": round(float(1 - env_seen.mean()), 4)}, {}
        for name, predicted in predictions.items():
            s = score(y, np.eye(N)[predicted])
            entry[name] = {"macro_f1": s["macro_f1"], "accuracy": s["accuracy"], "per_class": per_class(np.array(s["confusion_matrix"], dtype=float))}
            per_group[name], group_ids = confusion_by_group(y, predicted, group)
            if name in LINES:   # the rebuilt lines must equal the scores A.1 saved
                reference = saved[column]["baselines"]["test"][name]
                assert (s["accuracy"], s["macro_f1"]) == (reference["accuracy"], reference["macro_f1"]), f"{column} {name}: {s['accuracy'], s['macro_f1']} != {reference}"
        group_class = np.array([y[group == g][0] for g in group_ids])
        values = bootstrap(per_group, stratified_draws(group_class, BOOTSTRAPS, rng))
        for name in predictions:
            replicates[name].append(values[name]["macro_f1"])
        result["splits"][column] = entry
        print(f"{column}: model {entry['model']['macro_f1']:.3f} / {entry['model']['accuracy']:.3f}, majority {entry['majority_class']['macro_f1']:.3f} / {entry['majority_class']['accuracy']:.3f}, "
              f"room lookup {entry['environment_lookup']['macro_f1']:.3f} / {entry['environment_lookup']['accuracy']:.3f}, unseen-room share {entry['share_of_windows_in_unseen_room']:.2f}")

    names = ["model"] + LINES
    summary = {}
    for name in names:
        rows = [result["splits"][f"split_{k}"][name] for k in range(N_SPLITS)]
        summary[name] = {metric: {"mean": round(float(np.mean([r[metric] for r in rows])), 3), "std": round(float(np.std([r[metric] for r in rows], ddof=1)), 3)} for metric in ("macro_f1", "accuracy")}
        summary[name]["per_class_mean"] = {c: {m: round(float(np.mean([r["per_class"][c][m] for r in rows if r["per_class"][c][m] is not None])), 3) for m in ("recall", "f1")} for c in CLASSES}
        summary[name]["macro_f1_range_95_of_the_mean"] = interval(np.mean(replicates[name], axis=0))
    for line in LINES:
        gap = np.mean(replicates["model"], axis=0) - np.mean(replicates[line], axis=0)
        summary[f"model_minus_{line}"] = {"macro_f1": round(summary["model"]["macro_f1"]["mean"] - summary[line]["macro_f1"]["mean"], 3), "macro_f1_range_95": interval(gap),
                                          "accuracy": round(summary["model"]["accuracy"]["mean"] - summary[line]["accuracy"]["mean"], 3), "share_of_resamples_model_better": round(float((gap > 0).mean()), 3)}
    result["summary"] = summary
    return result


def markdown(result: dict) -> str:
    s = result["summary"]
    lines = [f"# {CANDIDATES[result['main_model']]} against the two reference lines (test rows)\n",
             "| | Macro-F1 (mean ± std) | Range (95%) | Accuracy (mean ± std) | " + " | ".join(f"{c} recall" for c in CLASSES) + " |", "|---|---|---|---|" + "---|" * len(CLASSES)]
    for name, label in [("model", "Main model"), *[(l, NAMES[l]) for l in LINES]]:
        r = s[name]
        lines.append(f"| {label} | {r['macro_f1']['mean']:.3f} ± {r['macro_f1']['std']:.3f} | {r['macro_f1_range_95_of_the_mean']} | {r['accuracy']['mean']:.3f} ± {r['accuracy']['std']:.3f} | "
                     + " | ".join(f"{r['per_class_mean'][c]['recall']:.2f}" for c in CLASSES) + " |")
    lines += ["", "## Gap of the main model to each line (3-split mean)\n", "| Line | Macro-F1 gap | 95% range | Accuracy gap | Share of resamples where the model is better |", "|---|---|---|---|---|"]
    for l in LINES:
        g = s[f"model_minus_{l}"]
        lines.append(f"| {NAMES[l]} | {g['macro_f1']:+.3f} | {g['macro_f1_range_95']} | {g['accuracy']:+.3f} | {g['share_of_resamples_model_better']:.0%} |")
    lines += ["", "## Per class F1 (3-split mean)\n", "| | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
    for name, label in [("model", "Main model"), *[(l, NAMES[l]) for l in LINES]]:
        lines.append(f"| {label} | " + " | ".join(f"{s[name]['per_class_mean'][c]['f1']:.2f}" for c in CLASSES) + " |")
    lines += ["", "## Per split\n", "| Split | Main model macro-F1 / accuracy | Most frequent class | Room lookup | Windows in a room unseen in training |", "|---|---|---|---|---|"]
    for split, e in result["splits"].items():
        lines.append(f"| {split} | {e['model']['macro_f1']:.3f} / {e['model']['accuracy']:.3f} | {e['majority_class']['macro_f1']:.3f} / {e['majority_class']['accuracy']:.3f} ({e['majority_class_name']}) | "
                     f"{e['environment_lookup']['macro_f1']:.3f} / {e['environment_lookup']['accuracy']:.3f} | {e['share_of_windows_in_unseen_room']:.0%} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    out = run()
    (EVALUATION_DIR / "baseline_comparison.json").write_text(json.dumps(out, indent=1))
    (EVALUATION_DIR / "baseline_comparison.md").write_text(markdown(out))
    print()
    print(markdown(out))
