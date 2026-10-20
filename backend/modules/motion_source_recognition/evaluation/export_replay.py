"""Writes the replay JSON that the website shows for this module: backend/modules/motion_source_recognition/outputs/replay.json and the copy bundled with the frontend.

Everything in it comes from the result files of the pipeline (outputs/prep, outputs/models, outputs/experiments, outputs/evaluation) and from the real detector; nothing is typed in by hand
except the labels and the short limits. Blocks:
  frames          the frames the other pages read (home page, fusion): five invented demonstration frames (config.json, website.landing_demo), not results of the detector
                  (the preview of this module's page is dashboard.preview_frames: answered by inference.Detector (split 0), six consecutive windows of one validation recording
                  of each class (random recording, seed 42), then six made-up damaged windows (3 degraded, 3 invalid) so that every kind of output is shown. Each preview frame also
                  has `truth` (the true class, or Made-up) and, for the made-up ones, `note`, for the demo only (not part of the interface))
  preprocessing   a table of the data (the tiles of the data are in the dashboard block: the shared page does not lay out tiles on the stage cards)
  training        validation scores of the five trained models (the table view; the page itself uses dashboard.training_view: what was tuned, what was chosen, how the fit developed)
  experiments     one line per comparison that was made
  metrics         test scores of the models and the two reference lines, and the confusion matrix of the main model (sum of the three test splits)
  dashboard       the numbers behind the charts of the module page (models, rooms by class, outcomes by class, confusion, trade-off curve, rooms, speed, limits, one training view per model, the feature demo); the page is
                  frontend/src/modules/motion_source_recognition/index.jsx with its charts in viz/

Run from the repository root (needs the dataset location, see the README, and all earlier results including evaluation.training_curves):
    backend/modules/motion_source_recognition/.venv/bin/python -m backend.modules.motion_source_recognition.evaluation.export_replay
"""
import json

import numpy as np
import pandas as pd

from backend.shared.export import write_replay

from ..inference import Detector
from ..preprocessing.config import CLASSES, EVALUATION, EVALUATION_DIR, EXPERIMENTS_DIR, MODELS, MODELS_DIR, MODULE, PREP_DIR, ROUTE_LABELS, TRAINING, WEBSITE, data_root
from ..preprocessing.windows import read_windows
from ..preprocessing.splits import N_SPLITS
from ..training.summary import collect
from .model_anatomy import K as ANATOMY_SPLIT

EXP = EXPERIMENTS_DIR
FRAMES_PER_CLASS, STEP_SECONDS, SEED = WEBSITE["frames_per_class"], WEBSITE["step_seconds"], TRAINING["seed"]
MADE_UP = WEBSITE["made_up_label"]
LABELS = {name: m["label"] for name, m in MODELS.items()}


def features_of(model: str) -> str:
    """'MiniRocket (9,996)': the feature route of the model and the number of features it was trained on."""
    run = json.loads((MODELS_DIR / model / "split_0" / "run.json").read_text())
    return f"{ROUTE_LABELS[MODELS[model]['route']]} ({run['features']:,})"


J = lambda path: json.loads(path.read_text())
pm = lambda mean, std, d=3: f"{mean:.{d}f} ± {std:.{d}f}"


def frames() -> list:
    manifest, splits = pd.read_csv(PREP_DIR / "manifest.csv"), pd.read_csv(PREP_DIR / "splits.csv")
    val = manifest[manifest["id"].isin(splits.loc[splits["split_0"] == "val", "id"])]
    rng = np.random.default_rng(SEED)
    detector, out, t, base = Detector(split=0), [], 0, None
    for c in CLASSES:
        groups = val.loc[val["label"] == c, "group_id"].unique()
        group = rng.choice(groups)
        rows = val[val["group_id"] == group].sort_values("session").head(FRAMES_PER_CLASS)
        for window in read_windows([str(data_root() / p) for p in rows["file_path"]]):
            if base is None:
                base = window
            out.append({"t": t, **detector(window), "truth": c})
            t += STEP_SECONDS
    for note, window in made_up_windows(base):   # made-up bad windows, answered by the same detector, so that every kind of output of the module is shown
        out.append({"t": t, **detector(window), "truth": MADE_UP, "note": note})
        t += STEP_SECONDS
    return out


def landing_frames() -> list:
    """The `frames` of the replay, which the home page and the fusion page read: invented demonstration frames (config.json, website.landing_demo), not results of the detector.
    The home page walks through the union of the time stamps of all modules, so they use the same ones as the other modules' demonstration data: a fan at first, then a person stays."""
    return [{"t": f["t"], "label": f["label"], "confidence": max(f["probs"].values()), "probs": f["probs"], "quality": {"ok": True, "level": "ok", "reasons": []}} for f in WEBSITE["landing_demo"]]


def made_up_windows(base: np.ndarray) -> list:
    """A real window, damaged in six ways: three of the quality level `degraded` (dead channels, a constant channel, values out of range) and three of `invalid` (a missing value, too few steps, too few channels)."""
    dead, flat, wild, hole = base.copy(), base.copy(), base.copy(), base.copy()
    dead[10:18] = 0
    flat[30] = flat[30].mean()
    wild[50] += 20
    hole[100, 200] = np.nan
    return [("8 channels are dead (all zero)", dead), ("one channel is constant", flat), ("one channel has values out of range", wild),
            ("one value is missing", hole), ("the window is too short (300 steps, not 500)", base[:, :300]), ("channels are missing (100, not 232)", base[:100])]


def preprocessing() -> tuple:
    m = pd.read_csv(PREP_DIR / "manifest.csv")
    per_class = [{"Class": c, "Windows": f"{(m.label == c).sum():,}", "Individuals": int(m[m.label == c].user.nunique()), "Recordings": int(m[m.label == c].group_id.nunique()),
                  "Rooms": int(m[m.label == c].environment.nunique())} for c in CLASSES]
    tiles = [{"label": "WINDOWS", "value": f"{len(m):,}"}, {"label": "RECORDINGS", "value": int(m.group_id.nunique())}, {"label": "ROOMS", "value": int(m.environment.nunique())}]
    return {"table": per_class}, tiles


def training() -> dict:
    rows = collect()
    out = []
    for model in LABELS:
        mine = [r for r in rows if r["model"] == model]
        f1, acc = [r["macro_f1"] for r in mine], [r["accuracy"] for r in mine]
        out.append({"Model": LABELS[model], "Features": features_of(model), "Validation macro-F1": pm(np.mean(f1), np.std(f1, ddof=1)), "Validation accuracy": pm(np.mean(acc), np.std(acc, ddof=1)),
                    "Training time (s, per split)": int(np.mean([r["training_seconds"] for r in mine])), "Model size (MB)": round(max(r["model_megabytes"] for r in mine), 1)})
    return {"table": out}


def experiments() -> dict:
    mean = lambda values: float(np.mean(values))
    fair, comp = J(EXP / "fair_comparison" / "fair_comparison.json"), J(EXP / "complementary" / "complementary.json")
    models, smooth, phys = J(EXP / "model_comparison.json"), J(EXP / "temporal_smoothing.json"), J(EXP / "physical_evaluation.json")
    spec = J(EXP / "spectral" / "spectral_evaluation.json")
    summ = {r["model"]: [] for r in collect()}
    for r in collect():
        summ[r["model"]].append(r["macro_f1"])
    lr_stat = mean([fair["a1"][f"split_{k}"]["validation"]["macro_f1"] for k in range(N_SPLITS)])
    extra = J(EVALUATION_DIR / "test_scores_extra.json")["models"]
    conf = J(EVALUATION_DIR / "confidence_threshold.json")["splits"]
    e11 = phys["in_one_home_E11"]
    f3 = lambda v: f"{v:.3f}"
    rf_s, lr_s, rf_m, lr_m = mean(summ["random_forest"]), lr_stat, mean(summ["random_forest_minirocket"]), mean(summ["logistic_regression"])
    soft = mean([models[f"split_{k}"]["models"]["soft_voting"]["macro_f1"] for k in range(N_SPLITS)])
    boost = mean([comp["gb"][f"split_{k}"]["macro_f1"] for k in range(N_SPLITS)])
    window = EVALUATION["smoothing_windows"]
    sm1 = mean([smooth["logistic_regression"]["n1"][f"split_{k}"]["macro_f1"] for k in range(N_SPLITS)])
    sm6 = mean([smooth["logistic_regression"][f"n{window}"][f"split_{k}"]["macro_f1"] for k in range(N_SPLITS)])
    ece_before, ece_after = mean([v["before"]["ece"] for v in conf.values()]), mean([v["after"]["ece"] for v in conf.values()])
    r_stat, r_phys = e11["statistics | Random Forest"]["recall"], e11["physical v2 | Random Forest"]["recall"]
    room_acc = phys["leave_one_room_out"]["physical v2 | Random Forest"]["accuracy"]
    ex = {m: extra[m]["macro_f1"]["mean"] for m in ("random_forest_statistics_physical", "logistic_regression_statistics_physical", "logistic_regression")}
    # name, verdict tone, verdict label, short result, question (table view), then four short lists: why it was tried, what was done, what was found, what follows. **text** is highlighted.
    rows = [
        ("Features or algorithm?", "good", "Features matter most", f"MiniRocket {rf_m:.2f} to {lr_m:.2f}, statistics {lr_s:.2f} to {rf_s:.2f}",
         "Random Forest and Logistic Regression on statistics or MiniRocket features (validation macro-F1)",
         ["The first two models differed in both algorithm and features", "Which of the two explains the gap?"],
         ["2 × 2 test: **Random Forest** and **Logistic Regression**, each on **statistics** and on **MiniRocket**", "Same windows, validation macro-F1, mean of 3 splits"],
         [f"Random Forest: statistics **{f3(rf_s)}**, MiniRocket **{f3(rf_m)}**", f"Logistic Regression: statistics **{f3(lr_s)}**, MiniRocket **{f3(lr_m)}**"],
         ["The **features** matter more than the algorithm", "Logistic Regression on MiniRocket is the best, and became the main model"]),
        ("Average two models", "neutral", "No gain", f"{f3(soft)} against {f3(lr_m)}", "Does averaging the Random Forest and the Logistic Regression help?",
         ["The two kinds of model may make different mistakes", "Averaging could then correct them"],
         ["Averaged the **probabilities** of the Random Forest and the Logistic Regression"],
         [f"Averaged: **{f3(soft)}**", f"Logistic Regression alone: **{f3(lr_m)}**"],
         ["**No gain**: the average is no better", "Kept the single, smaller model"]),
        ("Gradient boosting", "neutral", "No gain", f"macro-F1 {f3(boost)}", "Does a different kind of model make different mistakes?",
         ["A different kind of model may make different mistakes"],
         ["Gradient boosting on the statistics features", "Also averaged with the other models"],
         [f"Macro-F1 **{f3(boost)}**", "Averaged with the others: nothing added"],
         ["**No gain**: not used"]),
        (f"Average over {window} windows", "good", "Small gain", f"{f3(sm1)} to {f3(sm6)}", "Does smoothing the probabilities over time help?",
         ["A window is 5 s; the source rarely changes between two windows"],
         [f"Averaged the probabilities of the last **{window}** windows of the same run", "Compared with no averaging"],
         [f"Macro-F1 **{f3(sm1)} → {f3(sm6)}**", "No harm to the other numbers"],
         ["**Small gain**, so it is **optional**", "Price: a change of source shows up a few windows later"]),
        ("Calibration", "good", "Helps", f"error {ece_before:.3f} to {ece_after:.3f}", "Does the confidence mean what it says? One temperature per split, validation rows.",
         ["The confidence decides **unknown**, so it must mean what it says"],
         ["Measured how far the confidence is from the real hit rate (calibration error)", "Divided the logits by **one temperature per split**, fitted on the validation rows"],
         [f"Calibration error **{ece_before:.3f} → {ece_after:.3f}**"],
         ["**Helps**: confidence can be compared between splits", "The unknown rule is built on it"]),
        ("Spectral-shape features", "neutral", "No gain", "one antenna link scores within 0.01 of all four", "Do finer frequency features help?",
         ["Finer frequency features might separate the classes better"],
         ["Added spectral-shape numbers per channel (band shares, entropy, flatness, centre of the spectrum)", "Also tried **one antenna link** instead of four"],
         ["**No gain** over MiniRocket", "One link scores within **0.01** of all four"],
         ["**No gain**: not used"]),
        ("Physical features, one home", "warn", "One home only", f"robot recall {r_stat['IRobot']:.2f} to {r_phys['IRobot']:.2f}",
         "Physically motivated numbers (spread over channels, periodicity, dynamics), room E11, other recordings of the same room",
         ["Numbers with a physical meaning might make the rare devices easier to recognise, and easier to explain"],
         ["About 50 numbers: spread over channels, periodicity, 20 to 50 Hz content, bursts, dynamics", "Random Forest, room **E11**, tested on other recordings of the same room"],
         [f"Robot recall **{r_stat['IRobot']:.2f} → {r_phys['IRobot']:.2f}**", f"Human {r_phys['Human']:.2f}, Pet {r_phys['Pet']:.2f}"],
         ["Works **in one home**", "Designed after looking at this room, so **optimistic**"]),
        ("Physical features, new room", "neutral", "No gain", f"accuracy {room_acc:.2f}", "The same, with one whole room held out",
         ["Does the gain in E11 hold in a room the model has not seen?"],
         ["Same features, with each whole room held out in turn"],
         [f"Accuracy **{room_acc:.2f}**, about chance"],
         ["**No gain**", "The problem is the change of room and people, not the features"]),
        ("Statistics + physical", "warn", "Comparable, optimistic", f"{ex['random_forest_statistics_physical']:.2f} and {ex['logistic_regression_statistics_physical']:.2f} against {ex['logistic_regression']:.2f}",
         "Interpretable numbers added to the statistics features (test rows)",
         ["Can the interpretable numbers join the statistics features and match the main model?"],
         ["Random Forest and Logistic Regression on statistics + physical", "Scored on the test rows, after the model choice was frozen"],
         [f"Macro-F1 **{ex['random_forest_statistics_physical']:.2f}** and **{ex['logistic_regression_statistics_physical']:.2f}**", f"Main model: **{ex['logistic_regression']:.2f}**"],
         ["**Comparable**, not better", "Scores are **optimistic**: the features were designed after seeing the whole dataset", "The model choice is unchanged"]),
    ]
    plain = lambda items: "; ".join(i.replace("**", "") for i in items)
    return {"table": [{"Experiment": a, "Question": e, "Result": plain(found)} for a, _, _, _, e, _, _, found, _ in rows],
            "cards": [{"name": a, "tone": b, "verdict": c, "result": d, "why": w, "did": did, "found": found, "so": so} for a, b, c, d, _, w, did, found, so in rows]}


def metrics() -> dict:
    scores, base, err = J(EVALUATION_DIR / "test_scores_extra.json"), J(EVALUATION_DIR / "baseline_comparison.json")["summary"], J(EVALUATION_DIR / "error_analysis.json")
    main = J(EVALUATION_DIR / "model_selection.json")["main_model"]
    table = []
    for model, e in scores["models"].items():
        table.append({"Model": LABELS[model] + (" (chosen)" if model == main else "") + (" *" if MODELS[model].get("optimistic") else ""), "Macro-F1": pm(e["macro_f1"]["mean"], e["macro_f1"]["std"]),
                      "Accuracy": pm(e["accuracy"]["mean"], e["accuracy"]["std"]), **{f"{c} recall": f"{e['recall_mean'][c]:.2f}" for c in CLASSES}})
    for key, name in (("environment_lookup", "Reference: the most frequent class of the room"), ("majority_class", "Reference: always the most frequent class")):
        r = base[key]
        table.append({"Model": name, "Macro-F1": pm(r["macro_f1"]["mean"], r["macro_f1"]["std"]), "Accuracy": pm(r["accuracy"]["mean"], r["accuracy"]["std"]),
                      **{f"{c} recall": f"{r['per_class_mean'][c]['recall']:.2f}" for c in CLASSES}})
    matrix = np.sum([np.array(s["confusion_matrix"]) for s in err["splits"].values()], axis=0).tolist()
    return {"table": table, "confusion": {"labels": CLASSES, "matrix": matrix}}


PARAMETERS = {   # what each setting does; tuned = searched on the validation rows, fixed = the same for every split
    "forest": [{"name": "Smallest leaf", "tuned": True, "where": "Bottom of every tree", "effect": "How small a group a tree may split further. Small: detailed trees. Large: simpler trees."},
               {"name": "Trees", "tuned": False, "where": "The vote", "effect": "How many trees vote. More trees give a steadier answer."},
               {"name": "Features per split", "tuned": False, "where": "Every box of a tree", "effect": "Each box looks at only a few random features, so the trees differ from each other."},
               {"name": "Class weights", "tuned": False, "where": "Counting windows", "effect": "Rare classes count more, so Fan and IRobot are not ignored."}],
    "linear": [{"name": "Regularisation C", "tuned": True, "where": "The weights", "effect": "Small C keeps the weights small (simple model). Large C lets them grow to fit the training windows."},
               {"name": "Iterations", "tuned": False, "where": "The training loop", "effect": "Each iteration improves the weights a little. Training stops when they stop changing, or at 300."},
               {"name": "Standardisation", "tuned": False, "where": "Before the formulas", "effect": "Puts every feature on the same scale, so none wins just by having big numbers."},
               {"name": "Class weights", "tuned": False, "where": "The loss", "effect": "A mistake on a rare class costs more, so Fan and IRobot are not ignored."}]}


def training_view() -> list:
    """For each of the five models: what was tuned, what was chosen, and how the fit developed (evaluation/training_curves.py). All on the validation rows."""
    curves = J(EVALUATION_DIR / "training_curves.json")
    anatomy = J(EVALUATION_DIR / "model_anatomy.json")
    fair = J(EXP / "fair_comparison" / "fair_comparison.json")["a2"]
    text = {"random_forest": ("300 trees vote. Each tree learns from random windows and random features.", "min_samples_leaf", "Smallest leaf", "The smallest group of windows a tree may keep together. Small: detailed trees. Large: smoother trees."),
            "linear": ("One weight per feature, added up into probabilities. The weights are adjusted step by step until they stop changing.", "C", "Regularisation C", "Small C keeps the weights small (a simpler model); large C fits the training windows more closely.")}
    out = []
    for model in LABELS:
        kind = "linear" if model.startswith("logistic") else "forest"
        blurb, pname, plabel, phelp = text["linear" if kind == "linear" else "random_forest"]
        tuning, chosen, fits, curve = {}, {}, {}, {}
        for k in range(N_SPLITS):
            split = f"split_{k}"
            run = J(MODELS_DIR / model / split / "run.json")
            chosen[split] = run["chosen_parameters"][pname]
            if model == "random_forest_minirocket":   # the leaf sizes were searched in the fair comparison; the saved run holds only the refit
                rows = [{"value": t["value"], "macro_f1": t["macro_f1"], "accuracy": t["accuracy"], "seconds": t["fit_seconds"]} for t in fair[split]["tried"]]
            else:
                rows = [{"value": t["parameters"][pname], "macro_f1": t["macro_f1"], "accuracy": t["accuracy"], "seconds": t["fit_seconds"],
                         **({"iterations": t["iterations"], "converged": t["converged"]} if kind == "linear" else {})} for t in run["tuning"]]
            seen, unique = set(), []
            for r in sorted(rows, key=lambda r: r["value"]):
                if r["value"] not in seen:
                    seen.add(r["value"])
                    unique.append(r)
            tuning[split] = unique
            fits[split] = {"train_windows": run["train_windows"], "validation_macro_f1": run["validation"]["macro_f1"], "validation_accuracy": run["validation"]["accuracy"], "features": run["features"]}
            if kind == "linear":
                c = curves["linears"][model][split]
                curve[split] = c["points"]
                fits[split].update({"converged": c["converged"], "iterations": c["points"][-1]["iterations"]})
            else:
                curve[split] = curves["forests"][model][split]
        size = max(r["model_megabytes"] for r in collect() if r["model"] == model)
        out.append({"key": model, "name": LABELS[model], "kind": kind, "features": features_of(model), "blurb": blurb, "param": {"name": pname, "label": plabel, "help": phelp, "log": True},
                    "tuning": tuning, "chosen": chosen, "curve": curve, "fits": fits, "megabytes": round(size, 1), "optimistic": bool(MODELS[model].get("optimistic")), "anatomy": {**anatomy[model], "split": f"split_{ANATOMY_SPLIT}"}, "parameters": PARAMETERS[kind]})
    return out


def dashboard(data_tiles: list) -> dict:
    scores_extra, thr = J(EVALUATION_DIR / "test_scores_extra.json"), J(EVALUATION_DIR / "confidence_threshold_test.json")
    scores = scores_extra["models"]["logistic_regression"]
    room, lat, one = J(EVALUATION_DIR / "leave_one_room_out.json"), J(EVALUATION_DIR / "latency.json"), J(EVALUATION_DIR / "single_room_E11.json")
    err, base, curve = J(EVALUATION_DIR / "error_analysis.json"), J(EVALUATION_DIR / "baseline_comparison.json")["summary"], J(EVALUATION_DIR / "confidence_threshold.json")["splits"]
    m = pd.read_csv(PREP_DIR / "manifest.csv")
    S = thr["summary"]
    chosen = J(EVALUATION_DIR / "model_selection.json")["main_model"]
    val = {mm: [r for r in collect() if r["model"] == mm] for mm in LABELS}
    models_chart = []
    for mm in LABELS:
        t = scores_extra["models"][mm]
        v_f1, v_acc = [r["macro_f1"] for r in val[mm]], [r["accuracy"] for r in val[mm]]
        models_chart.append({"name": LABELS[mm], "features": features_of(mm), "chosen": mm == chosen, "optimistic": bool(MODELS[mm].get("optimistic")),
                             "validation": {"macro_f1": [float(np.mean(v_f1)), float(np.std(v_f1, ddof=1))], "accuracy": [float(np.mean(v_acc)), float(np.std(v_acc, ddof=1))]},
                             "test": {"macro_f1": [t["macro_f1"]["mean"], t["macro_f1"]["std"]], "accuracy": [t["accuracy"]["mean"], t["accuracy"]["std"]]},
                             "seconds": int(np.mean([r["training_seconds"] for r in val[mm]])), "megabytes": round(max(r["model_megabytes"] for r in val[mm]), 1)})
    references = [{"name": name, "test": {"macro_f1": [base[k]["macro_f1"]["mean"], base[k]["macro_f1"]["std"]], "accuracy": [base[k]["accuracy"]["mean"], base[k]["accuracy"]["std"]]}}
                  for k, name in (("environment_lookup", "Reference: most frequent class of the room"), ("majority_class", "Reference: always the most frequent class"))]
    rooms = sorted(m.environment.unique())
    room_class = [{"room": r, "cells": [{"class": c, "windows": int(((m.environment == r) & (m.label == c)).sum()), "individuals": int(m[(m.environment == r) & (m.label == c)].user.nunique())} for c in CLASSES]} for r in rooms]
    # what happens to 100 windows of each class: right, answered as another class, answered unknown (per split, then the mean)
    outcome = []
    for c in CLASSES:
        spl = list(thr["splits"].values())
        no_rule_right = float(np.mean([e["recall_all"][c] for e in spl]))
        u = [e["rejected_by_true_class"][c] for e in spl]
        r = [(1 - e["rejected_by_true_class"][c]) * e["recall_kept"][c] for e in spl]
        w = [(1 - e["rejected_by_true_class"][c]) * (1 - e["recall_kept"][c]) for e in spl]
        outcome.append({"class": c, "without_rule": {"right": no_rule_right, "wrong": 1 - no_rule_right, "unknown": 0.0},
                        "with_rule": {"right": float(np.mean(r)), "wrong": float(np.mean(w)), "unknown": float(np.mean(u))}})
    matrix = np.sum([np.array(s["confusion_matrix"]) for s in err["splits"].values()], axis=0).tolist()
    n_rows = len(curve["split_0"]["trade_off_after_calibration"])
    tradeoff = [{"share": curve["split_0"]["trade_off_after_calibration"][i]["rejected_share"], **{key: float(np.mean([curve[s]["trade_off_after_calibration"][i][key] for s in curve]))
                for key in ("accuracy_kept", "macro_f1_kept", "errors_rejected")}} for i in range(n_rows)]
    compare = [{"name": "Test rows: rooms and people seen", "value": scores["accuracy"]["mean"], "detail": "accuracy, 3 test splits"},
               {"name": "One room, other recordings", "value": one["accuracy"], "detail": "room E11: trained and tested on different recordings"},
               {"name": "New room and new people", "value": room["pooled"]["accuracy"], "detail": "every room left out in turn, all rooms together"}]
    rooms_acc = [{"room": r, "accuracy": e["accuracy"], "windows": e["windows"], "classes": list(e["classes_in_room"]), "recall": e["recall"]} for r, e in room["rooms"].items()]
    unknown = [{"split": s, "temperature": thr["frozen"][s]["temperature"], "threshold": thr["frozen"][s]["threshold"], "unknown": e["rejected_share"],
                "accuracy_all": e["accuracy_all"], "accuracy_answered": e["accuracy_kept"]} for s, e in thr["splits"].items()]
    speed = [{"name": LABELS[mm], "ms": c["total_ms"]["median_ms"], "megabytes": c["model_megabytes"] + c["also_ship_megabytes"]} for mm, c in lat["candidates"].items()]
    limits = [
        {"tone": "critical", "title": "New rooms and people: close to chance", "detail": f"Rooms, people and pets seen in training are recognised well; with a whole room held out the accuracy is {room['pooled']['accuracy']:.2f}, like guessing among four classes."},
        {"tone": "critical", "title": "IRobot and Fan: not recognised on new devices", "detail": "Test recall 0.00 to 0.01; most of their windows become unknown (43% to 93%), the rest are answered as Human or Pet."},
        {"tone": "warn", "title": "Most mistakes are confident ones", "detail": f"Calibration lowers the calibration error from {S['ece_before']['mean']:.3f} to {S['ece_after']['mean']:.3f} on the test rows; it does not remove it."},
        {"tone": "info", "title": "Unknown is not an unknown object", "detail": "It only means that the highest probability is below the threshold."},
        {"tone": "info", "title": "Not tested with the other modules", "detail": "Their data share no room with this one, so the interface is an agreement, not an end-to-end test."},
        {"tone": "info", "title": "Scores marked * are optimistic", "detail": "Their features were designed after looking at the whole dataset."},
    ]
    fields = [{"field": "label", "meaning": "Human, Pet, IRobot, Fan or unknown"}, {"field": "probs", "meaning": "calibrated probability of each class"},
              {"field": "confidence", "meaning": "the largest probability"}, {"field": "quality", "meaning": "ok, degraded or invalid, with the reasons"}]
    rooms_per_individual = m.groupby(["label", "user"]).environment.nunique()
    one_room = {c: (int((rooms_per_individual[c] == 1).sum()), int(len(rooms_per_individual[c]))) for c in CLASSES}
    insight = (f"Almost every individual was recorded in one room only: {one_room['Human'][0]} of {one_room['Human'][1]} people, {one_room['Pet'][0]} of {one_room['Pet'][1]} pets, "
               f"{one_room['IRobot'][0]} of {one_room['IRobot'][1]} robots, {one_room['Fan'][0]} of {one_room['Fan'][1]} fans. A new room therefore also means new individuals.")
    return {"n_splits": N_SPLITS, "data_tiles": data_tiles, "insight": insight, "models": models_chart, "references": references, "room_class": room_class, "outcome": outcome, "confusion": {"labels": CLASSES, "matrix": matrix},
            "training_view": training_view(), "feature_demo": J(EVALUATION_DIR / "feature_demo.json"), "tradeoff": tradeoff, "tradeoff_chosen_share": thr["budget"], "compare": compare, "rooms": rooms_acc, "unknown_rule": unknown, "speed": speed, "limits": limits, "fields": fields}


def run() -> dict:
    prep, tiles = preprocessing()
    preview = frames()
    payload = {"module": MODULE, "version": "1.0", "mock": False, "frames": landing_frames(), "preprocessing": prep, "training": training(), "experiments": experiments(),
               "metrics": metrics(), "dashboard": {**dashboard(tiles), "preview_frames": preview}}
    write_replay(MODULE, payload, bundle=True)
    return payload


if __name__ == "__main__":
    out = run()
    print(f"{len(out['frames'])} frames for the other pages, {len(out['dashboard']['preview_frames'])} for the preview; blocks: {[k for k in out if k not in ('module', 'version', 'mock', 'frames')]}")
    print([(f["t"], f["label"], f["confidence"]) for f in out["frames"]])
