"""A.3 step 2: score the three candidates on the TEST rows, once.

Order of work (each step stops the run if it fails):
  1. check that the saved test probabilities still match the fingerprints recorded in step 1 (evaluation/model_selection.py)
  2. count the test windows and recording groups per class and split (no model involved)
  3. score every candidate on every split with the same function as the validation scores (training/common.score)
  4. the same with the probabilities averaged over the last 6 windows of a run (extra row; the main numbers are without smoothing)
  5. uncertainty: the recording groups of each split are resampled with replacement 1,000 times, separately inside each class (so every class stays
     present; the same draws for all models, so differences are paired), and the macro-F1 of every resample is recomputed; the 95% range is the 2.5th to
     97.5th percentile. The unit is the group, not the window, because the windows of one group are near copies. A class with a single test group
     (IRobot and Fan in every split) cannot vary in this scheme, so the ranges show only the uncertainty that comes from the other classes.
     (A first version resampled all groups together; a rare class then vanished from about a third of the resamples and the ranges were misleading.)
  6. validation and test side by side

The scores are written once. The script refuses to run again unless --again is given, and the file records how often it was run.

Run from the repository root:
    backend/modules/motion_source_recognition/.venv/bin/python -m backend.modules.motion_source_recognition.evaluation.test_scoring
Writes outputs/evaluation/test_scores.json and test_scores.md.
"""
import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..experiments.temporal_smoothing import smooth
from ..preprocessing.config import CLASSES, EVALUATION, EVALUATION_DIR, MODELS, PREP_DIR, TRAINING
from ..preprocessing.splits import N_SPLITS
from ..training.common import CLASS_INDEX, MODELS_DIR, score
from .model_selection import CANDIDATES, fingerprint

FILE = EVALUATION_DIR / "test_scores.json"
EXTRA = {name: m["label"] + " *" for name, m in MODELS.items() if m.get("optimistic")}   # scored after the choice was frozen; * = optimistic
EXTRA_NOTE = ("\\* Added after the model choice was frozen. The physical features were designed after looking at the recordings of the whole dataset (the E11 robot and fan recordings, and the "
              "differences between classes over all rooms), which include windows of the validation and test sets, so the scores of these two models are more optimistic than the others'. "
              "The frozen choice is not changed by them.")
BOOTSTRAPS = EVALUATION["bootstraps"]
SEED = TRAINING["seed"]
SMOOTHING_WINDOWS = EVALUATION["smoothing_windows"]
N = len(CLASSES)


def confusion_by_group(y: np.ndarray, predicted: np.ndarray, group: np.ndarray) -> tuple:
    """Confusion counts (rows = true class, columns = predicted) of every group, as a (groups, N * N) array, and the group ids."""
    ids, inverse = np.unique(group, return_inverse=True)
    counts = np.zeros((len(ids), N * N))
    np.add.at(counts, (inverse, y * N + predicted), 1)
    return counts, ids


def stratified_draws(group_class: np.ndarray, resamples: int, rng: np.random.Generator) -> np.ndarray:
    """(resamples, groups) weights: the groups of each class are resampled with replacement among themselves, so no class can disappear."""
    draws = np.zeros((resamples, len(group_class)))
    for c in np.unique(group_class):
        members = np.flatnonzero(group_class == c)
        draws[:, members] = rng.multinomial(len(members), np.full(len(members), 1.0 / len(members)), size=resamples)
    return draws


def macro_f1_present(confusion: np.ndarray) -> float:
    """Macro-F1 from a confusion matrix, averaged over the classes that have at least one true window."""
    tp, true, pred = np.diag(confusion), confusion.sum(axis=1), confusion.sum(axis=0)
    present = true > 0
    f1 = np.where(true + pred > 0, 2 * tp / np.maximum(true + pred, 1), 0.0)
    return float(f1[present].mean())


def bootstrap(per_group: dict, draws: np.ndarray) -> dict:
    """per_group: model -> (groups, N*N) counts. draws: (B, groups) multinomial weights. Returns per model the macro-F1 and accuracy of every resample."""
    out = {}
    for model, counts in per_group.items():
        matrices = (draws @ counts).reshape(len(draws), N, N)
        out[model] = {"macro_f1": np.array([macro_f1_present(m) for m in matrices]),
                      "accuracy": np.array([np.trace(m) / m.sum() for m in matrices])}
    return out


def interval(values: np.ndarray) -> list:
    return [round(float(np.percentile(values, 2.5)), 3), round(float(np.percentile(values, 97.5)), 3)]


def run(extra: dict | None = None) -> dict:
    selection = json.loads((EVALUATION_DIR / "model_selection.json").read_text())
    labels = {**CANDIDATES, **(extra or {})}
    models = list(labels)
    for model in CANDIDATES:   # step 1; the extra models are not part of the frozen choice
        for k in range(N_SPLITS):
            if fingerprint(model, k) != selection["candidates"][model]["test_probability_fingerprint"][f"split_{k}"]:
                raise SystemExit(f"{model} split_{k}: the saved test probabilities changed after the choice was recorded; stop")
    print("fingerprints match")

    splits, manifest = pd.read_csv(PREP_DIR / "splits.csv"), pd.read_csv(PREP_DIR / "manifest.csv").set_index("id")
    for k in range(N_SPLITS):   # step 2, before any scoring: every class must be present in every test set
        present = set(splits.loc[splits[f"split_{k}"] == "test", "label"])
        if present != set(CLASSES):
            raise SystemExit(f"split_{k}: no test window of {sorted(set(CLASSES) - present)}; nothing was scored, decide how to treat the missing class first")
    rng = np.random.default_rng(SEED)
    result = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "fingerprints_checked": True, "main_model": selection["main_model"], "labels": labels,
              "extra_models_test_fingerprints": {m: {f"split_{k}": fingerprint(m, k) for k in range(N_SPLITS)} for m in (extra or {})},
              "bootstrap": {"resamples": BOOTSTRAPS, "seed": SEED, "unit": "recording group", "stratified_by_class": True,
                                                                            "note": "classes with one test group (IRobot, Fan) cannot vary; ranges reflect Human and Pet groups only"}, "test_counts": {}, "models": {m: {"splits": {}} for m in models}}
    replicates = {m: {"macro_f1": [], "accuracy": []} for m in models}
    for k in range(N_SPLITS):
        rows = splits[splits[f"split_{k}"] == "test"]   # step 2
        y_all = rows["label"].map(CLASS_INDEX).to_numpy()
        group_all = rows["group_id"].to_numpy()
        counts = {c: {"windows": int((y_all == i).sum()), "groups": int(len(np.unique(group_all[y_all == i])))} for i, c in enumerate(CLASSES)}
        result["test_counts"][f"split_{k}"] = counts
        print(f"split_{k} test windows and groups:", {c: (v["windows"], v["groups"]) for c, v in counts.items()})
        per_group, probas = {}, {}
        for model in models:   # step 3
            z = np.load(MODELS_DIR / model / f"split_{k}" / "probabilities.npz", allow_pickle=False)
            if not (z["ids_test"] == rows["id"].to_numpy()).all():
                raise SystemExit(f"{model} split_{k}: test rows are not the rows of splits.csv")
            proba = z["proba_test"]
            meta = manifest.loc[z["ids_test"]]
            smoothed = score(y_all, smooth(proba, meta["group_id"].to_numpy(), meta["session"].to_numpy(), SMOOTHING_WINDOWS))   # step 4
            s = score(y_all, proba)
            s["macro_f1_present_classes"] = round(macro_f1_present(np.array(s["confusion_matrix"], dtype=float)), 4)
            s["with_smoothing_over_6_windows"] = {"macro_f1": smoothed["macro_f1"], "accuracy": smoothed["accuracy"], "recall": {c: v["recall"] for c, v in smoothed["per_class"].items()}}
            run_json = json.loads((MODELS_DIR / model / f"split_{k}" / "run.json").read_text())["validation"]
            s["validation"] = {"macro_f1": run_json["macro_f1"], "accuracy": run_json["accuracy"]}
            result["models"][model]["splits"][f"split_{k}"] = s
            per_group[model], group_ids = confusion_by_group(y_all, proba.argmax(axis=1), group_all)
        group_class = np.array([y_all[group_all == g][0] for g in group_ids])
        draws = stratified_draws(group_class, BOOTSTRAPS, rng)   # step 5
        for model, values in bootstrap(per_group, draws).items():
            for metric in ("macro_f1", "accuracy"):
                replicates[model][metric].append(values[metric])
            result["models"][model]["splits"][f"split_{k}"]["macro_f1_range_95"] = interval(values["macro_f1"])

    main = selection["main_model"]
    for model in models:   # step 5 and 6: over the splits
        entry = result["models"][model]
        per_split = [entry["splits"][f"split_{k}"] for k in range(N_SPLITS)]
        for metric in ("macro_f1", "accuracy"):
            values = [s[metric] for s in per_split]
            mean_replicates = np.mean(replicates[model][metric], axis=0)
            entry[metric] = {"mean": round(float(np.mean(values)), 3), "std": round(float(np.std(values, ddof=1)), 3), "range_95_of_the_mean": interval(mean_replicates)}
        entry["recall_mean"] = {c: round(float(np.mean([s["per_class"][c]["recall"] for s in per_split])), 3) for c in CLASSES}
        entry["macro_f1_with_smoothing_mean"] = round(float(np.mean([s["with_smoothing_over_6_windows"]["macro_f1"] for s in per_split])), 3)
        entry["validation_macro_f1_mean"] = round(float(np.mean([s["validation"]["macro_f1"] for s in per_split])), 3)
        if model != main:
            difference = np.mean(replicates[main]["macro_f1"], axis=0) - np.mean(replicates[model]["macro_f1"], axis=0)
            entry["main_minus_this_macro_f1"] = {"mean": round(float(np.mean([result["models"][main]["splits"][f"split_{k}"]["macro_f1"] - entry["splits"][f"split_{k}"]["macro_f1"] for k in range(N_SPLITS)])), 3),
                                                 "range_95": interval(difference), "share_of_resamples_main_is_better": round(float((difference > 0).mean()), 3)}
    return result


def markdown(result: dict) -> str:
    models, main, labels = list(result["models"]), result["main_model"], result.get("labels", CANDIDATES)
    lines = ["# Test scores (scored once)\n", f"Created {result['created']}. Fingerprints of the saved test probabilities were checked first. Main model: **{labels[main]}**.\n",
             "## Test windows and recording groups\n", "| Split | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
    for split, counts in result["test_counts"].items():
        lines.append(f"| {split} | " + " | ".join(f"{counts[c]['windows']} windows, {counts[c]['groups']} groups" for c in CLASSES) + " |")
    lines += ["", "## Test scores (mean ± standard deviation over the 3 splits; range = 95% of group resamples within each class, of the 3-split mean; IRobot and Fan have one test group per split, so the range cannot show their uncertainty)\n",
              "| Model | Macro-F1 | Range | Accuracy | Range | " + " | ".join(f"{c} recall" for c in CLASSES) + " | With smoothing (macro-F1) | Validation macro-F1 |", "|---|---|---|---|---|" + "---|" * len(CLASSES) + "---|---|"]
    for m in models:
        e = result["models"][m]
        lines.append(f"| {labels[m]} | {e['macro_f1']['mean']:.3f} ± {e['macro_f1']['std']:.3f} | {e['macro_f1']['range_95_of_the_mean']} | {e['accuracy']['mean']:.3f} ± {e['accuracy']['std']:.3f} | {e['accuracy']['range_95_of_the_mean']} | "
                     + " | ".join(f"{e['recall_mean'][c]:.2f}" for c in CLASSES) + f" | {e['macro_f1_with_smoothing_mean']:.3f} | {e['validation_macro_f1_mean']:.3f} |")
    lines += ["", "## Per split: macro-F1 (95% range) and the two rare classes\n", "| Model | Split | Macro-F1 | Range | Accuracy | IRobot precision / recall | Fan precision / recall |", "|---|---|---|---|---|---|---|"]
    for m in models:
        for split, s in result["models"][m]["splits"].items():
            ir, fan = s["per_class"]["IRobot"], s["per_class"]["Fan"]
            lines.append(f"| {labels[m]} | {split} | {s['macro_f1']:.3f} | {s['macro_f1_range_95']} | {s['accuracy']:.3f} | {ir['precision']:.2f} / {ir['recall']:.2f} | {fan['precision']:.2f} / {fan['recall']:.2f} |")
    lines += ["", "## Main model minus each other candidate (macro-F1, 3-split mean)\n", "| Compared with | Difference | 95% range | Share of resamples where the main model is better |", "|---|---|---|---|"]
    for m in models:
        d = result["models"][m].get("main_minus_this_macro_f1")
        if d:
            lines.append(f"| {labels[m]} | {d['mean']:+.3f} | {d['range_95']} | {d['share_of_resamples_main_is_better']:.0%} |")
    if any(m in EXTRA for m in models):
        lines += ["", EXTRA_NOTE]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--again", action="store_true", help="score again although the scores file exists (the count is recorded)")
    parser.add_argument("--with-extra", action="store_true", help="score the extra models (statistics + physical) next to the three frozen candidates; writes test_scores_extra.*")
    args = parser.parse_args()
    target, name = (EVALUATION_DIR / "test_scores_extra.json", "test_scores_extra") if args.with_extra else (FILE, "test_scores")
    runs = 0
    if target.exists():
        if not args.again:
            raise SystemExit(f"{target.name} exists: the test rows were scored already; run with --again only if you mean to score a second time")
        runs = json.loads(target.read_text()).get("times_scored", 1)
    scores = run(EXTRA if args.with_extra else None)
    scores["times_scored"] = runs + 1
    target.write_text(json.dumps(scores, indent=1))
    (EVALUATION_DIR / f"{name}.md").write_text(markdown(scores))
    print(markdown(scores))
