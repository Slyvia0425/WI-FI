"""A.3 step 7: a confidence threshold for the main model, so that it can answer "unknown" instead of guessing.

The confidence of a window is the highest of the four class probabilities. A window whose confidence is below the threshold is answered "unknown" (a decision state, not a fifth
class). Because the Logistic Regression is over-confident in some splits and flat in others (evaluation of step 4), the probabilities are first calibrated with the simplest method:
one temperature T per split, fitted on the validation rows only (the logits are divided by T; the predicted class never changes).

Stage 1 (this version): calibration and the trade-off curve, on the VALIDATION rows only; the test rows are not read.
  - T per split, and the calibration error (ECE) and the reliability table before and after, on the validation rows (in sample: one number is fitted)
  - for rejecting the least confident 0 to 50% of the windows: the threshold it means, the accuracy and macro-F1 of the windows that are kept, the share of the errors
    that are rejected, and the share of every class that is rejected
Stage 2 (--stage 2): the budget chosen from the stage 1 curve is 20%, i.e. the threshold of each split is the 20th percentile of the calibrated confidences of its validation rows
(answer "unknown" for the least confident fifth of the windows). T and the threshold are written to confidence_threshold_frozen.json BEFORE any test row is read; then the test rows are
reported once: every test window belongs to a known class, so the share answered "unknown" is the false-rejection rate of known classes. The script refuses to run twice unless --again.

Stage 3 (--stage 3): the rule the user chose after seeing the stage 2 report, one threshold PER CLASS from a precision target: for each class, the lowest threshold such that, among the validation
windows called that class and kept, at least 90% really are that class, with at least 30 windows kept; a class that cannot reach this (IRobot and Fan in every split but one) is never
claimed: its windows are answered "unknown". The "lean" (the class with the highest probability) is always given next to the label. The thresholds are written to
confidence_threshold_per_class_frozen.json before the test rows are read, and the test rows are reported a SECOND time (the first report, stage 2, used one threshold for all classes and is kept).

Run from the repository root:
    backend/modules/motion_source_recognition/.venv/bin/python -m backend.modules.motion_source_recognition.evaluation.confidence_threshold [--stage 2|3]
Stage 1 writes outputs/evaluation/confidence_threshold.json and .md; stage 2 writes confidence_threshold_frozen.json, confidence_threshold_test.json and .md; stage 3 writes confidence_threshold_per_class_frozen.json, confidence_threshold_per_class_test.json and .md.
"""
import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from ..calibration import reliability, temper, to_logits
from ..preprocessing.config import CLASSES, EVALUATION, EVALUATION_DIR, PREP_DIR
from ..preprocessing.splits import N_SPLITS
from ..training.common import CLASS_INDEX, MODELS_DIR, score


BUDGET = EVALUATION["unknown_budget"]   # chosen from the stage 1 curve: the share of the windows answered "unknown"
SHARES = EVALUATION["unknown_shares"]   # share of the windows answered "unknown"


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """The T that minimises the negative log-likelihood of the true class on these rows (T above 1 flattens, below 1 sharpens)."""
    def nll(log_t):
        p = temper(logits, float(np.exp(log_t)))
        return -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
    return float(np.exp(minimize_scalar(nll, bounds=(-3.0, 5.0), method="bounded").x))


def trade_off(proba: np.ndarray, y: np.ndarray, shares: list = SHARES) -> list:
    """For every share: reject the least confident windows of that share; what remains."""
    confidence, predicted = proba.max(axis=1), proba.argmax(axis=1)
    wrong = predicted != y
    rows = []
    for share in shares:
        threshold = float(np.quantile(confidence, share)) if share > 0 else 0.0
        keep = confidence >= threshold if share > 0 else np.ones(len(y), dtype=bool)
        kept = score(y[keep], np.eye(len(CLASSES))[predicted[keep]]) if keep.any() else None
        rows.append({"rejected_share": round(1 - float(keep.mean()), 3), "threshold": round(threshold, 4), "accuracy_kept": kept["accuracy"], "macro_f1_kept": kept["macro_f1"],
                     "errors_rejected": round(float(1 - wrong[keep].sum() / max(wrong.sum(), 1)), 3),
                     "rejected_by_true_class": {c: round(float(1 - keep[y == i].mean()), 3) for i, c in enumerate(CLASSES) if (y == i).any()}})
    return rows


def stage_one() -> dict:
    splits = pd.read_csv(PREP_DIR / "splits.csv").set_index("id")
    selection = json.loads((EVALUATION_DIR / "model_selection.json").read_text())
    model = selection["main_model"]
    result = {"model": model, "stage": "calibration and trade-off curve on the validation rows; no test row read", "splits": {}}
    for k in range(N_SPLITS):
        z = np.load(MODELS_DIR / model / f"split_{k}" / "probabilities.npz", allow_pickle=False)
        y = splits.loc[z["ids_val"], "label"].map(CLASS_INDEX).to_numpy()
        logits = to_logits(z["proba_val"])
        temperature = fit_temperature(logits, y)
        calibrated = temper(logits, temperature)
        assert (calibrated.argmax(axis=1) == z["proba_val"].argmax(axis=1)).all(), "the temperature changed a prediction"
        result["splits"][f"split_{k}"] = {"temperature": round(temperature, 3), "windows": int(len(y)),
                                          "before": reliability(z["proba_val"], y), "after": reliability(calibrated, y),
                                          "trade_off_before_calibration": trade_off(z["proba_val"], y), "trade_off_after_calibration": trade_off(calibrated, y)}
    return result


def markdown(r: dict) -> str:
    L = ["# Confidence threshold of the main model: calibration and trade-off (validation rows only)\n", "| Split | Temperature T | ECE before | ECE after | Windows with confidence >= 0.99 before / after |", "|---|---|---|---|---|"]
    for split, e in r["splits"].items():
        top = lambda rel: next((b["share"] for b in rel["bins"] if b["confidence"] == ">=0.99"), 0.0)
        L.append(f"| {split} | {e['temperature']} | {e['before']['ece']} | {e['after']['ece']} | {top(e['before']):.0%} / {top(e['after']):.0%} |")
    L += ["", "Reject the least confident windows (calibrated confidence); accuracy / macro-F1 of the windows that are kept, share of all errors that are rejected:", "",
          "| Rejected | " + " | ".join(f"{s} accuracy / macro-F1 / errors rejected" for s in r["splits"]) + " |", "|---|" + "---|" * len(r["splits"])]
    for i, share in enumerate(SHARES):
        cells = []
        for e in r["splits"].values():
            row = e["trade_off_after_calibration"][i]
            cells.append(f"{row['accuracy_kept']:.3f} / {row['macro_f1_kept']:.3f} / {row['errors_rejected']:.0%}")
        L.append(f"| {share:.0%} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


FROZEN, TEST_FILE = EVALUATION_DIR / "confidence_threshold_frozen.json", EVALUATION_DIR / "confidence_threshold_test.json"


def freeze(budget: float = BUDGET) -> dict:
    """T and the threshold of every split, from the validation rows only; written to disk before the test rows are read."""
    splits = pd.read_csv(PREP_DIR / "splits.csv").set_index("id")
    model = json.loads((EVALUATION_DIR / "model_selection.json").read_text())["main_model"]
    frozen = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": model, "budget_share_unknown_on_validation": budget,
              "rule": "unknown when the calibrated highest probability is below the threshold; calibrated = softmax(log p / T); threshold = quantile `budget` of the calibrated confidences of the validation rows",
              "splits": {}}
    for k in range(N_SPLITS):
        z = np.load(MODELS_DIR / model / f"split_{k}" / "probabilities.npz", allow_pickle=False)
        y = splits.loc[z["ids_val"], "label"].map(CLASS_INDEX).to_numpy()
        logits = to_logits(z["proba_val"])
        temperature = fit_temperature(logits, y)
        frozen["splits"][f"split_{k}"] = {"temperature": round(temperature, 4), "threshold": round(float(np.quantile(temper(logits, temperature).max(axis=1), budget)), 4)}
    return frozen


def report_split(proba: np.ndarray, y: np.ndarray, temperature: float, threshold: float) -> dict:
    calibrated = temper(to_logits(proba), temperature)
    confidence, predicted = calibrated.max(axis=1), calibrated.argmax(axis=1)
    keep, wrong = confidence >= threshold, predicted != y
    everything = score(y, np.eye(len(CLASSES))[predicted])
    kept = score(y[keep], np.eye(len(CLASSES))[predicted[keep]])
    return {"windows": int(len(y)), "rejected_share": round(1 - float(keep.mean()), 4), "accuracy_all": everything["accuracy"], "macro_f1_all": everything["macro_f1"],
            "accuracy_kept": kept["accuracy"], "macro_f1_kept": kept["macro_f1"], "recall_all": {c: v["recall"] for c, v in everything["per_class"].items()},
            "recall_kept": {c: v["recall"] for c, v in kept["per_class"].items()},
            "errors_rejected": round(float(1 - wrong[keep].sum() / max(wrong.sum(), 1)), 3), "rejected_that_were_wrong": round(float(wrong[~keep].mean()), 3) if (~keep).any() else None,
            "rejected_by_true_class": {c: round(float(1 - keep[y == i].mean()), 3) for i, c in enumerate(CLASSES)},
            "confident_errors_kept": int((wrong & keep).sum()), "ece_before": reliability(proba, y)["ece"], "ece_after": reliability(calibrated, y)["ece"],
            "reliability_after": reliability(calibrated, y)["bins"]}


def stage_two(budget: float = BUDGET) -> dict:
    frozen = freeze(budget)
    FROZEN.write_text(json.dumps(frozen, indent=1))   # before any test row is read
    splits = pd.read_csv(PREP_DIR / "splits.csv").set_index("id")
    result = {"model": frozen["model"], "budget": budget, "frozen": frozen["splits"], "splits": {}}
    for k in range(N_SPLITS):
        z = np.load(MODELS_DIR / frozen["model"] / f"split_{k}" / "probabilities.npz", allow_pickle=False)
        y = splits.loc[z["ids_test"], "label"].map(CLASS_INDEX).to_numpy()
        f = frozen["splits"][f"split_{k}"]
        result["splits"][f"split_{k}"] = report_split(z["proba_test"], y, f["temperature"], f["threshold"])
    mean_std = lambda key: {"mean": round(float(np.mean([e[key] for e in result["splits"].values()])), 3), "std": round(float(np.std([e[key] for e in result["splits"].values()], ddof=1)), 3)}
    result["summary"] = {key: mean_std(key) for key in ("rejected_share", "accuracy_all", "accuracy_kept", "macro_f1_all", "macro_f1_kept", "errors_rejected", "ece_before", "ece_after")}
    return result


def markdown_test(r: dict) -> str:
    S = r["summary"]
    L = [f"# Confidence threshold on the test rows (budget {r['budget']:.0%} of the validation windows answered \"unknown\"; frozen before the test rows were read)\n",
         "| Split | T | Threshold | Rejected (false rejection of known classes) | Accuracy: all to kept | Macro-F1: all to kept | Errors rejected | Rejected windows that were wrong |", "|---|---|---|---|---|---|---|---|"]
    for split, e in r["splits"].items():
        f = r["frozen"][split]
        L.append(f"| {split} | {f['temperature']} | {f['threshold']} | {e['rejected_share']:.1%} | {e['accuracy_all']:.3f} to {e['accuracy_kept']:.3f} | {e['macro_f1_all']:.3f} to {e['macro_f1_kept']:.3f} | {e['errors_rejected']:.0%} | "
                 + (f"{e['rejected_that_were_wrong']:.0%}" if e["rejected_that_were_wrong"] is not None else "-") + " |")
    L.append(f"| **mean ± std** | | | {S['rejected_share']['mean']:.1%} ± {S['rejected_share']['std']:.1%} | {S['accuracy_all']['mean']:.3f} to {S['accuracy_kept']['mean']:.3f} | {S['macro_f1_all']['mean']:.3f} to {S['macro_f1_kept']['mean']:.3f} | {S['errors_rejected']['mean']:.0%} | |")
    L += ["", "Share of each true class answered \"unknown\":", "", "| Split | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
    L += [f"| {split} | " + " | ".join(f"{e['rejected_by_true_class'][c]:.0%}" for c in CLASSES) + " |" for split, e in r["splits"].items()]
    L += ["", "Recall of the kept windows (all windows in brackets):", "", "| Split | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
    L += [f"| {split} | " + " | ".join(f"{e['recall_kept'][c]:.2f} ({e['recall_all'][c]:.2f})" for c in CLASSES) + " |" for split, e in r["splits"].items()]
    L += ["", f"Calibration on the test rows: ECE {S['ece_before']['mean']:.3f} before, {S['ece_after']['mean']:.3f} after (mean of the splits). Kept windows that are still wrong: "
          + ", ".join(f"{split} {e['confident_errors_kept']}" for split, e in r["splits"].items()) + "."]
    return "\n".join(L) + "\n"


PRECISION_TARGET, MIN_KEPT = 0.90, 30
PC_FROZEN, PC_TEST_FILE = EVALUATION_DIR / "confidence_threshold_per_class_frozen.json", EVALUATION_DIR / "confidence_threshold_per_class_test.json"


def class_threshold(confidence: np.ndarray, right: np.ndarray, target: float = PRECISION_TARGET, min_kept: int = MIN_KEPT) -> float | None:
    """For the windows called one class: the lowest threshold such that the windows at or above it are right at least `target` of the time, with at least `min_kept` of them; None if there is none."""
    if len(confidence) < min_kept:
        return None
    order = np.argsort(-confidence)
    precision = np.cumsum(right[order]) / np.arange(1, len(order) + 1)
    ok = np.flatnonzero((precision >= target) & (np.arange(1, len(order) + 1) >= min_kept))
    return float(confidence[order][ok.max()]) if len(ok) else None


def decide_per_class(proba: np.ndarray, thresholds: dict) -> tuple:
    """(label index or -1 for unknown, lean index, confidence). A class without a threshold (None) is never claimed."""
    lean, confidence = proba.argmax(axis=1), proba.max(axis=1)
    limit = np.array([np.inf if thresholds[i] is None else thresholds[i] for i in range(len(CLASSES))])
    return np.where(confidence >= limit[lean], lean, -1), lean, confidence


def freeze_per_class() -> dict:
    splits = pd.read_csv(PREP_DIR / "splits.csv").set_index("id")
    base = json.loads(FROZEN.read_text())   # the temperatures of stage 2 (validation only)
    frozen = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": base["model"], "precision_target": PRECISION_TARGET, "min_windows_kept": MIN_KEPT,
              "rule": "label = the class with the highest calibrated probability if that probability is at least the threshold of that class, else unknown; a class with threshold null is never claimed; lean = that class always",
              "splits": {}}
    for k in range(N_SPLITS):
        z = np.load(MODELS_DIR / base["model"] / f"split_{k}" / "probabilities.npz", allow_pickle=False)
        y = splits.loc[z["ids_val"], "label"].map(CLASS_INDEX).to_numpy()
        T = base["splits"][f"split_{k}"]["temperature"]
        p = temper(to_logits(z["proba_val"]), T)
        pred, conf = p.argmax(axis=1), p.max(axis=1)
        thresholds = {c: class_threshold(conf[pred == i], y[pred == i] == i) for i, c in enumerate(CLASSES)}
        frozen["splits"][f"split_{k}"] = {"temperature": T, "thresholds": {c: (None if t is None else round(t, 4)) for c, t in thresholds.items()},
                                          "validation_windows_called": {c: int((pred == i).sum()) for i, c in enumerate(CLASSES)}}
    return frozen


def report_split_per_class(proba: np.ndarray, y: np.ndarray, temperature: float, thresholds: dict) -> dict:
    p = temper(to_logits(proba), temperature)
    label, lean, confidence = decide_per_class(p, {i: thresholds[c] for i, c in enumerate(CLASSES)})
    keep = label >= 0
    wrong = lean != y
    everything = score(y, np.eye(len(CLASSES))[lean])
    kept = score(y[keep], np.eye(len(CLASSES))[lean[keep]]) if keep.any() else None
    claimed = {c: int(((label == i)).sum()) for i, c in enumerate(CLASSES)}
    return {"windows": int(len(y)), "unknown_share": round(1 - float(keep.mean()), 4), "accuracy_all": everything["accuracy"], "macro_f1_all": everything["macro_f1"],
            "accuracy_kept": kept["accuracy"] if kept else None, "macro_f1_kept": kept["macro_f1"] if kept else None,
            "recall_kept": {c: v["recall"] for c, v in kept["per_class"].items()} if kept else None,
            "claimed_windows": claimed, "share_of_claims_that_are_right": {c: (round(float(((label == i) & (y == i)).sum() / claimed[c]), 3) if claimed[c] else None) for i, c in enumerate(CLASSES)},
            "unknown_by_true_class": {c: round(float((label[y == i] < 0).mean()), 3) for i, c in enumerate(CLASSES)},
            "errors_turned_into_unknown": round(float(1 - wrong[keep].sum() / max(wrong.sum(), 1)), 3), "unknown_that_were_wrong": round(float(wrong[~keep].mean()), 3) if (~keep).any() else None,
            "wrong_claims_kept": int((wrong & keep).sum()), "lean_equals_true_class_when_unknown": round(float((lean[~keep] == y[~keep]).mean()), 3) if (~keep).any() else None}


def stage_three() -> dict:
    frozen = freeze_per_class()
    PC_FROZEN.write_text(json.dumps(frozen, indent=1))   # before any test row is read
    splits = pd.read_csv(PREP_DIR / "splits.csv").set_index("id")
    result = {"model": frozen["model"], "precision_target": PRECISION_TARGET, "frozen": frozen["splits"], "splits": {},
              "note": "second report on the test rows: the first used one threshold for all classes (confidence_threshold_test.json); this rule was chosen after seeing it"}
    for k in range(N_SPLITS):
        z = np.load(MODELS_DIR / frozen["model"] / f"split_{k}" / "probabilities.npz", allow_pickle=False)
        y = splits.loc[z["ids_test"], "label"].map(CLASS_INDEX).to_numpy()
        f = frozen["splits"][f"split_{k}"]
        result["splits"][f"split_{k}"] = report_split_per_class(z["proba_test"], y, f["temperature"], f["thresholds"])
    first = json.loads(TEST_FILE.read_text())
    mean_std = lambda src, key: {"mean": round(float(np.mean([e[key] for e in src.values()])), 3), "std": round(float(np.std([e[key] for e in src.values()], ddof=1)), 3)}
    result["summary"] = {key: mean_std(result["splits"], key) for key in ("unknown_share", "accuracy_all", "accuracy_kept", "macro_f1_all", "macro_f1_kept", "errors_turned_into_unknown")}
    result["first_report_global_threshold"] = {"unknown_share": mean_std(first["splits"], "rejected_share"), "accuracy_kept": mean_std(first["splits"], "accuracy_kept"), "macro_f1_kept": mean_std(first["splits"], "macro_f1_kept")}
    return result


def markdown_per_class(r: dict) -> str:
    S, F = r["summary"], r["first_report_global_threshold"]
    L = [f"# Per-class confidence thresholds on the test rows (precision target {r['precision_target']:.0%}; frozen before the test rows were read; second report)\n",
         "Thresholds (calibrated probability; \"never\" = the class is never claimed, its windows are answered unknown):\n", "| Split | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
    for split, f in r["frozen"].items():
        L.append(f"| {split} | " + " | ".join("never" if f["thresholds"][c] is None else f"{f['thresholds'][c]:.3f}" for c in CLASSES) + " |")
    L += ["", "| Split | Unknown share | Accuracy: all to kept | Macro-F1: all to kept | Errors turned into unknown | Unknown windows that were wrong | Wrong claims kept |", "|---|---|---|---|---|---|---|"]
    for split, e in r["splits"].items():
        L.append(f"| {split} | {e['unknown_share']:.1%} | {e['accuracy_all']:.3f} to {e['accuracy_kept']:.3f} | {e['macro_f1_all']:.3f} to {e['macro_f1_kept']:.3f} | {e['errors_turned_into_unknown']:.0%} | "
                 + (f"{e['unknown_that_were_wrong']:.0%}" if e["unknown_that_were_wrong"] is not None else "-") + f" | {e['wrong_claims_kept']} |")
    L.append(f"| **mean ± std** | {S['unknown_share']['mean']:.1%} ± {S['unknown_share']['std']:.1%} | {S['accuracy_all']['mean']:.3f} to {S['accuracy_kept']['mean']:.3f} | {S['macro_f1_all']['mean']:.3f} to {S['macro_f1_kept']['mean']:.3f} | {S['errors_turned_into_unknown']['mean']:.0%} | | |")
    L += ["", f"First report (one threshold for all classes, 20% budget): unknown {F['unknown_share']['mean']:.1%}, accuracy of kept {F['accuracy_kept']['mean']:.3f}, macro-F1 of kept {F['macro_f1_kept']['mean']:.3f}.", "",
          "Share of each true class answered unknown:", "", "| Split | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
    L += [f"| {split} | " + " | ".join(f"{e['unknown_by_true_class'][c]:.0%}" for c in CLASSES) + " |" for split, e in r["splits"].items()]
    L += ["", "Windows claimed as each class, and the share of those claims that are right:", "", "| Split | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
    L += [f"| {split} | " + " | ".join(f"{e['claimed_windows'][c]} ({'-' if e['share_of_claims_that_are_right'][c] is None else format(e['share_of_claims_that_are_right'][c], '.0%')})" for c in CLASSES) + " |" for split, e in r["splits"].items()]
    L += ["", "When the answer is unknown, the lean (best guess) equals the true class in: " + ", ".join(f"{split} {e['lean_equals_true_class_when_unknown']:.0%}" for split, e in r["splits"].items() if e["lean_equals_true_class_when_unknown"] is not None) + " of the unknown windows."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--again", action="store_true", help="stage 2 only: report on the test rows a second time")
    args = parser.parse_args()
    if args.stage == 1:
        out = stage_one()
        (EVALUATION_DIR / "confidence_threshold.json").write_text(json.dumps(out, indent=1))
        (EVALUATION_DIR / "confidence_threshold.md").write_text(markdown(out))
        print(markdown(out))
    elif args.stage == 3:
        if PC_TEST_FILE.exists() and not args.again:
            raise SystemExit("the per-class rule was reported on the test rows already; run with --again only if you mean to report it a second time")
        out = stage_three()
        PC_TEST_FILE.write_text(json.dumps(out, indent=1))
        (EVALUATION_DIR / "confidence_threshold_per_class_test.md").write_text(markdown_per_class(out))
        print(markdown_per_class(out))
    else:
        if TEST_FILE.exists() and not args.again:
            raise SystemExit("the test rows were reported already; run with --again only if you mean to report a second time")
        out = stage_two()
        TEST_FILE.write_text(json.dumps(out, indent=1))
        (EVALUATION_DIR / "confidence_threshold_test.md").write_text(markdown_test(out))
        print(markdown_test(out))
