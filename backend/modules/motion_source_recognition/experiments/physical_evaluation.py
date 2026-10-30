"""Report-only experiment: do physically motivated features help, alone or added to the statistics and MiniRocket features?

Feature sets (per window); v3 is v2 plus 6 more raw-signal numbers (how the frequency content changes during the window, how fast the signal forgets itself, skewness and
kurtosis of the strongest component) and 2 numbers about the dominant frequencies of the channels, and "consistent" keeps only the v3 numbers whose class difference points the same way
in at least 75% of the rooms that contain both classes (mean |AUC - 0.5| of 0.10 or more, at least 2 rooms, computed in the training rooms only):
  statistics               the 2,320 statistics features (preprocessing/features.py)
  physical v1              23 numbers: motion size, dominant frequency, band share, spectral shape (from the statistics and spectral files, averaged over channels),
                           uniformity across channels and links, and 4 dynamics over the last 6 windows of the same recording
  physical v2              v1 + the first 24 raw-signal numbers of preprocessing/physical.py (spread over channels, periodicity, regularity, 20 to 50 Hz content, bursts) + 5 more dynamics
                           (over 24 windows, and over 6 windows of three of the new numbers): 52 numbers
  statistics + physical v2 / MiniRocket + physical v2   the physical v2 numbers appended (standardised) to the large sets

Two tests, both with models fixed in advance (Random Forest; Logistic Regression where stated):
  in one home    E11 (all four classes), five folds over recordings, as in evaluation/single_room.py
  leave one room out   every room held out in turn (as evaluation/leave_one_room_out.py, whose MiniRocket result was accuracy 0.248, macro-F1 0.159)

Selection bias: the physical features were designed after looking at the E11 robot and fan recordings. The E11 results are therefore optimistic; the other nine rooms
(including the other robot IR02 in E19 and the fans of E12 and E13) were not looked at.

Run from the repository root (after preprocessing.physical):
    nice -n 15 backend/modules/motion_source_recognition/.venv/bin/python -m backend.modules.motion_source_recognition.experiments.physical_evaluation [--skip-loro]
Writes outputs/experiments/physical_evaluation.json.
"""
import argparse
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..preprocessing.config import CLASSES, EXPERIMENTS_DIR, PREP_DIR
from ..preprocessing.physical import RAW_NAMES
from ..training.common import CLASS_INDEX, ROUTES, confusion, human_vs_rest
from ..training.rooms import prepare, summarise

EXP = EXPERIMENTS_DIR
OUT = EXP / "physical_evaluation.json"


def build() -> dict:
    """The physical route (outputs/prep/features_physical.npy) and the smaller sets cut from it by column name."""
    table = np.load(PREP_DIR / "features_physical.npy")
    columns = json.loads((PREP_DIR / "features_physical_columns.json").read_text())
    v1_columns = columns[:19] + ["dyn_std_cv", "dyn_std_change", "dyn_share_slow", "dyn_peak_to_mean"]
    v2_columns = [c for c in columns if c not in ("dominant_freq_spread", "dominant_freq_mode_share") and c not in RAW_NAMES[24:]]
    pick = lambda names: table[:, [columns.index(c) for c in names]]
    return {"physical v1": pick(v1_columns), "physical v2": pick(v2_columns), "physical v3": table, "columns_v2": v2_columns, "columns_v3": columns,
            "statistics": np.load(PREP_DIR / "features_stats.npy", mmap_mode="r")}


PAIRS = [("Human", "Pet"), ("IRobot", "Human"), ("IRobot", "Pet"), ("Fan", "Human"), ("Fan", "Pet"), ("IRobot", "Fan")]
MIN_WINDOWS, MIN_ROOMS, AGREE, EFFECT = 150, 2, 0.75, 0.10


def room_effects(X: np.ndarray, y: np.ndarray, rooms: np.ndarray) -> dict:
    """(class pair, room) -> AUC - 0.5 of every feature within that room (positive: the first class has the higher value)."""
    out = {}
    for room in sorted(set(rooms)):
        in_room = rooms == room
        for a, b in PAIRS:
            xa, xb = X[in_room & (y == CLASS_INDEX[a])], X[in_room & (y == CLASS_INDEX[b])]
            if len(xa) < MIN_WINDOWS or len(xb) < MIN_WINDOWS:
                continue
            ranks = rankdata(np.vstack([xa, xb]), axis=0)
            out[((a, b), room)] = (ranks[:len(xa)].sum(axis=0) - len(xa) * (len(xa) + 1) / 2) / (len(xa) * len(xb)) - 0.5
    return out


def consistent_features(effects: dict, n_features: int, exclude_room: str | None = None) -> tuple:
    """Features whose class difference has the same sign in at least AGREE of the rooms (outside `exclude_room`) holding both classes, with a mean effect of at least EFFECT."""
    keep, detail = np.zeros(n_features, dtype=bool), {}
    for pair in PAIRS:
        values = np.array([v for (p, r), v in effects.items() if p == pair and r != exclude_room])
        if len(values) < MIN_ROOMS:
            continue
        mean = values.mean(axis=0)
        ok = (np.sign(values) == np.sign(mean)).mean(axis=0) >= AGREE
        ok &= np.abs(mean) >= EFFECT
        keep |= ok
        detail[pair] = (ok, mean, len(values))
    return keep, detail


def make_model(kind: str, n_features: int, weights: dict, threads: int):
    if kind == "Logistic Regression":
        return LogisticRegression(C=1.0 if n_features < 200 else 0.1, max_iter=1000, class_weight=weights, random_state=42)
    large = n_features > 1000
    return RandomForestClassifier(n_estimators=100 if large else 300, min_samples_leaf=50 if large else 20, max_samples=0.5 if large else None, class_weight=weights, n_jobs=threads, random_state=42)


def fit_predict(kind: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, threads: int) -> np.ndarray:
    counts = np.bincount(y_train, minlength=len(CLASSES))
    weights = {i: float(len(y_train) / (len(CLASSES) * counts[i])) for i in range(len(CLASSES))}
    scaler = StandardScaler().fit(X_train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = make_model(kind, X_train.shape[1], weights, threads).fit(scaler.transform(X_train), y_train)
    return model.predict(scaler.transform(X_test))


def assemble(sets: dict, names: list, pos: np.ndarray, extra_path=None) -> np.ndarray:
    parts = [np.ascontiguousarray(sets[n][pos], dtype=np.float32) for n in names]
    return np.concatenate(parts, axis=1)


def in_one_home(sets: dict, threads: int) -> dict:
    rows, folds = prepare("E11")
    minirocket = np.load(PREP_DIR / ROUTES["minirocket"].format(k=0), mmap_mode="r")
    variants = {"physical v1": (["physical v1"], ["Random Forest", "Logistic Regression"]), "physical v2": (["physical v2"], ["Random Forest", "Logistic Regression"]),
                "statistics": (["statistics"], ["Random Forest"]), "statistics + physical v2": (["statistics", "physical v2"], ["Random Forest"]),
                "MiniRocket + physical v2": (["MiniRocket", "physical v2"], ["Logistic Regression"])}
    sets = {**sets, "MiniRocket": minirocket}
    results = {}
    for variant, (names, kinds) in variants.items():
        for kind in kinds:
            total, per_recording = np.zeros((len(CLASSES), len(CLASSES)), dtype=int), []
            for f, train, test in folds:
                predicted = fit_predict(kind, assemble(sets, names, train["pos"].to_numpy()), train["y"].to_numpy(), assemble(sets, names, test["pos"].to_numpy()), threads)
                np.add.at(total, (test["y"].to_numpy(), predicted), 1)
                for gid, idx in test.reset_index(drop=True).groupby("group_id").indices.items():
                    r = test.iloc[idx]
                    per_recording.append({"class": r["label"].iloc[0], "recording": r["activity"].iloc[0], "recall": round(float((predicted[idx] == r["y"].to_numpy()).mean()), 3)})
            results[f"{variant} | {kind}"] = summarise("E11", total, per_recording, [])
            r = results[f"{variant} | {kind}"]
            print(f"E11 {variant:26s} {kind:20s} accuracy {r['accuracy']:.3f}  recall " + "  ".join(f"{c} {r['recall'][c]:.2f}" for c in CLASSES), flush=True)
    return results


def leave_one_room_out(sets: dict, manifest: pd.DataFrame, threads: int) -> dict:
    quality = pd.read_csv(PREP_DIR / "quality.csv").set_index("id")
    hashes = quality.loc[manifest["id"], "content_hash"].to_numpy()
    y = manifest["label"].map(CLASS_INDEX).to_numpy()
    rooms = manifest["environment"].to_numpy()
    variants = {"physical v2": (["physical v2"], ["Random Forest", "Logistic Regression"]), "statistics": (["statistics"], ["Random Forest"]), "statistics + physical v2": (["statistics", "physical v2"], ["Random Forest"])}
    results = {}
    for variant, (names, kinds) in variants.items():
        full = assemble(sets, names, np.arange(len(manifest)))
        for kind in kinds:
            total, per_room = np.zeros((len(CLASSES), len(CLASSES)), dtype=int), {}
            for room in sorted(set(rooms)):
                held = rooms == room
                keep = ~held & ~np.isin(hashes, hashes[held])
                predicted = fit_predict(kind, full[keep], y[keep], full[held], threads)
                m = confusion(y[held], predicted)
                total += m
                per_room[room] = round(float(np.trace(m) / m.sum()), 3)
            tp, true, pred = np.diag(total), total.sum(axis=1), total.sum(axis=0)
            f1 = 2 * tp / np.maximum(true + pred, 1)
            yy = np.repeat(np.arange(len(CLASSES) * len(CLASSES)) // len(CLASSES), total.reshape(-1))
            pp = np.repeat(np.arange(len(CLASSES) * len(CLASSES)) % len(CLASSES), total.reshape(-1))
            results[f"{variant} | {kind}"] = {"accuracy": round(float(tp.sum() / total.sum()), 3), "macro_f1": round(float(f1.mean()), 3), "recall": {c: round(float(tp[i] / true[i]), 3) for i, c in enumerate(CLASSES)},
                                              "human_vs_rest_f1": human_vs_rest(yy, pp)["f1"], "accuracy_per_room": per_room}
            r = results[f"{variant} | {kind}"]
            print(f"leave-one-room-out {variant:26s} {kind:20s} accuracy {r['accuracy']:.3f}  macro-F1 {r['macro_f1']:.3f}  recall " + "  ".join(f"{c} {r['recall'][c]:.2f}" for c in CLASSES), flush=True)
    return results


def v3_in_one_home(sets: dict, manifest: pd.DataFrame, effects: dict, threads: int) -> dict:
    rows, folds = prepare("E11")
    keep, _ = consistent_features(effects, sets["physical v3"].shape[1], exclude_room="E11")
    names = np.array(sets["columns_v3"])
    print("consistent features selected without E11:", len(names[keep]), list(names[keep]), flush=True)
    results = {"selected_without_E11": list(names[keep])}
    for variant, columns in (("physical v3", slice(None)), ("physical v3 consistent", keep)):
        for kind in ("Random Forest", "Logistic Regression"):
            total, per_recording = np.zeros((len(CLASSES), len(CLASSES)), dtype=int), []
            for f, train, test in folds:
                Xtr, Xte = sets["physical v3"][train["pos"].to_numpy()][:, columns], sets["physical v3"][test["pos"].to_numpy()][:, columns]
                predicted = fit_predict(kind, Xtr, train["y"].to_numpy(), Xte, threads)
                np.add.at(total, (test["y"].to_numpy(), predicted), 1)
                for gid, idx in test.reset_index(drop=True).groupby("group_id").indices.items():
                    r = test.iloc[idx]
                    per_recording.append({"class": r["label"].iloc[0], "recording": r["activity"].iloc[0], "recall": round(float((predicted[idx] == r["y"].to_numpy()).mean()), 3)})
            results[f"{variant} | {kind}"] = summarise("E11", total, per_recording, [])
            r = results[f"{variant} | {kind}"]
            print(f"E11 {variant:24s} {kind:20s} accuracy {r['accuracy']:.3f}  recall " + "  ".join(f"{c} {r['recall'][c]:.2f}" for c in CLASSES), flush=True)
    return results


def v3_leave_one_room_out(sets: dict, manifest: pd.DataFrame, effects: dict, threads: int) -> dict:
    quality = pd.read_csv(PREP_DIR / "quality.csv").set_index("id")
    hashes = quality.loc[manifest["id"], "content_hash"].to_numpy()
    y, rooms, X = manifest["label"].map(CLASS_INDEX).to_numpy(), manifest["environment"].to_numpy(), sets["physical v3"]
    results = {}
    for variant in ("physical v3", "physical v3 consistent"):
        for kind in ("Random Forest", "Logistic Regression"):
            total, per_room, sizes = np.zeros((len(CLASSES), len(CLASSES)), dtype=int), {}, []
            for room in sorted(set(rooms)):
                held = rooms == room
                keep_rows = ~held & ~np.isin(hashes, hashes[held])
                columns = consistent_features(effects, X.shape[1], exclude_room=room)[0] if variant.endswith("consistent") else np.ones(X.shape[1], dtype=bool)
                if columns.sum() == 0:
                    columns = np.ones(X.shape[1], dtype=bool)
                sizes.append(int(columns.sum()))
                predicted = fit_predict(kind, X[keep_rows][:, columns], y[keep_rows], X[held][:, columns], threads)
                m = confusion(y[held], predicted)
                total += m
                per_room[room] = round(float(np.trace(m) / m.sum()), 3)
            tp, true, pred = np.diag(total), total.sum(axis=1), total.sum(axis=0)
            f1 = 2 * tp / np.maximum(true + pred, 1)
            yy, pp = np.repeat(np.arange(len(CLASSES) * len(CLASSES)) // len(CLASSES), total.reshape(-1)), np.repeat(np.arange(len(CLASSES) * len(CLASSES)) % len(CLASSES), total.reshape(-1))
            results[f"{variant} | {kind}"] = {"accuracy": round(float(tp.sum() / total.sum()), 3), "macro_f1": round(float(f1.mean()), 3), "recall": {c: round(float(tp[i] / true[i]), 3) for i, c in enumerate(CLASSES)},
                                              "human_vs_rest_f1": human_vs_rest(yy, pp)["f1"], "accuracy_per_room": per_room, "features_used_per_fold": sizes}
            r = results[f"{variant} | {kind}"]
            print(f"leave-one-room-out {variant:24s} {kind:20s} accuracy {r['accuracy']:.3f}  macro-F1 {r['macro_f1']:.3f}  recall " + "  ".join(f"{c} {r['recall'][c]:.2f}" for c in CLASSES) + f"  features {sizes}", flush=True)
    return results


def describe_consistency(effects: dict, columns: list) -> dict:
    """Over all rooms: for every class pair the features that point the same way in the rooms holding both classes (largest mean effects first)."""
    keep, detail = consistent_features(effects, len(columns))
    out = {}
    for pair, (ok, mean, n_rooms) in detail.items():
        order = [i for i in np.argsort(-np.abs(mean)) if ok[i]][:8]
        out["{} vs {}".format(*pair)] = {"rooms": n_rooms, "features": [(columns[i], round(float(mean[i]), 2)) for i in order]}
    return out


def importances(sets: dict, manifest: pd.DataFrame, threads: int) -> dict:
    """Which physical v2 numbers a Random Forest uses, in one home (E11) and over all rooms."""
    y = manifest["label"].map(CLASS_INDEX).to_numpy()
    X = sets["physical v2"]
    out = {}
    for name, mask in (("E11", (manifest["environment"] == "E11").to_numpy()), ("all rooms", np.ones(len(manifest), dtype=bool))):
        counts = np.bincount(y[mask], minlength=len(CLASSES))
        weights = {i: float(mask.sum() / (len(CLASSES) * counts[i])) for i in range(len(CLASSES))}
        model = RandomForestClassifier(n_estimators=150, min_samples_leaf=20, class_weight=weights, n_jobs=threads, random_state=42).fit(X[mask], y[mask])
        order = np.argsort(model.feature_importances_)[::-1][:10]
        out[name] = [(sets["columns_v2"][i], round(float(model.feature_importances_[i]), 3)) for i in order]
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-loro", action="store_true")
    parser.add_argument("--v3-only", action="store_true", help="only the v3 part; the earlier results in the file are kept")
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    manifest = pd.read_csv(PREP_DIR / "manifest.csv")
    sets = build()
    result = json.loads(OUT.read_text()) if (args.v3_only and OUT.exists()) else {}
    if not args.v3_only:
        result.update({"features_v2": sets["columns_v2"], "in_one_home_E11": in_one_home(sets, args.threads)})
        result["importances"] = importances(sets, manifest, args.threads)
        for name, top in result["importances"].items():
            print(f"most used physical v2 numbers ({name}):", top)
        if not args.skip_loro:
            result["leave_one_room_out"] = leave_one_room_out(sets, manifest, args.threads)
    else:
        y, rooms = manifest["label"].map(CLASS_INDEX).to_numpy(), manifest["environment"].to_numpy()
        effects = room_effects(sets["physical v3"], y, rooms)
        result["features_v3"] = sets["columns_v3"]
        result["consistency_all_rooms"] = describe_consistency(effects, sets["columns_v3"])
        for pair, d in result["consistency_all_rooms"].items():
            print(f"{pair} ({d['rooms']} rooms):", d["features"])
        result["in_one_home_E11_v3"] = v3_in_one_home(sets, manifest, effects, args.threads)
        if not args.skip_loro:
            result["leave_one_room_out_v3"] = v3_leave_one_room_out(sets, manifest, effects, args.threads)
    OUT.write_text(json.dumps(result, indent=1))
