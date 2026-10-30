"""A.3 step 4: where does the main model go wrong? Test rows, main model only (the frozen choice); no training, no new scoring of other models.

  1. confusion matrix per split, and the mean of the row-normalised matrices
  2. Human <-> Fan and Pet <-> Human confusion rates
  3. "is it a person?": Human against everything else, with the false alarms split by their true class
  4. error rate per room
  5. accuracy of every recording group of the rare classes, and the worst groups of the others
  6. confident mistakes: errors with a highest probability of at least 0.9
  7. same device or not (VALIDATION rows, three candidates): does a rare class do better when a recording of the same device is in the training rows?

The test probabilities are fingerprint-checked against the frozen choice before they are used.

Run from the repository root:
    backend/modules/motion_source_recognition/.venv/bin/python -m backend.modules.motion_source_recognition.evaluation.error_analysis
Writes outputs/evaluation/error_analysis.json and error_analysis.md.
"""
import json

import numpy as np
import pandas as pd

from ..preprocessing.config import CLASSES, EVALUATION, EVALUATION_DIR, PREP_DIR
from ..preprocessing.splits import N_SPLITS
from ..training.common import CLASS_INDEX, MODELS_DIR, confusion, human_vs_rest
from .model_selection import CANDIDATES, fingerprint

N = len(CLASSES)
CONFIDENT = EVALUATION["confident_threshold"]
HUMAN = CLASS_INDEX["Human"]


def row_normalised(matrix: np.ndarray) -> np.ndarray:
    total = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, total, out=np.zeros(matrix.shape, dtype=float), where=total > 0)


def confident_mistakes(proba: np.ndarray, y: np.ndarray) -> dict:
    confidence, predicted = proba.max(axis=1), proba.argmax(axis=1)
    wrong, sure = predicted != y, proba.max(axis=1) >= CONFIDENT
    return {"windows_at_least_0.9": round(float(sure.mean()), 3), "wrong_among_those": round(float((wrong & sure).sum() / max(sure.sum(), 1)), 3),
            "share_of_all_errors_that_are_confident": round(float((wrong & sure).sum() / max(wrong.sum(), 1)), 3),
            "share_of_correct_windows_below_0.9": round(float((~wrong & ~sure).sum() / max((~wrong).sum(), 1)), 3),
            "confident_errors_by_true_class": {c: int((wrong & sure & (y == i)).sum()) for i, c in enumerate(CLASSES)}}


def group_table(y: np.ndarray, predicted: np.ndarray, meta: pd.DataFrame) -> list:
    rows = []
    for gid, idx in meta.groupby("group_id").indices.items():
        called = np.bincount(predicted[idx], minlength=N)
        rows.append({"group_id": int(gid), "class": CLASSES[y[idx][0]], "user": meta["user"].iloc[idx[0]], "activity": meta["activity"].iloc[idx[0]], "room": meta["environment"].iloc[idx[0]],
                     "windows": int(len(idx)), "accuracy": round(float((predicted[idx] == y[idx]).mean()), 3), "most_called": CLASSES[int(called.argmax())], "called_share": round(float(called.max() / len(idx)), 3)})
    return rows


def room_table(y: np.ndarray, predicted: np.ndarray, meta: pd.DataFrame, train_rooms: set) -> list:
    rows = []
    for room, idx in meta.groupby("environment").indices.items():
        wrong = predicted[idx] != y[idx]
        rows.append({"room": room, "in_training": room in train_rooms, "windows": int(len(idx)), "classes": {c: int((y[idx] == i).sum()) for i, c in enumerate(CLASSES) if (y[idx] == i).any()},
                     "error_rate": round(float(wrong.mean()), 3), "errors": int(wrong.sum())})
    return sorted(rows, key=lambda r: -r["errors"])


def room_type_table(y: np.ndarray, predicted: np.ndarray, meta: pd.DataFrame, classes_in_room: dict) -> dict:
    """Error rate in rooms where the whole dataset has recordings of one class only, against rooms shared by several classes.

    Given for all windows and for Human and Pet windows only: every IRobot and Fan recording lies in a shared room and is almost always wrong,
    so the all-window figure for shared rooms mixes two things.
    """
    multi = meta["environment"].map(lambda room: len(classes_in_room[room]) > 1).to_numpy()
    wrong = predicted != y
    living = y <= 1   # Human, Pet
    return {scope: {kind: {"windows": int((mask & keep).sum()), "error_rate": round(float(wrong[mask & keep].mean()), 3) if (mask & keep).any() else None}
                    for kind, mask in (("rooms_with_one_class", ~multi), ("rooms_shared_by_several_classes", multi))}
            for scope, keep in (("all_windows", np.ones(len(y), dtype=bool)), ("human_and_pet_only", living))}


def same_device(manifest: pd.DataFrame, splits: pd.DataFrame, models: list) -> list:
    """Validation groups of the rare classes: is another recording of the same device (the same user id) in the training rows? Recall and F1 of the class in that split."""
    label_of = splits.set_index("id")["label"].map(CLASS_INDEX)
    rows = []
    for k in range(N_SPLITS):
        column = f"split_{k}"
        train_ids, val_ids = splits.loc[splits[column] == "train", "id"], splits.loc[splits[column] == "val", "id"]
        train_users = set(manifest[manifest["id"].isin(train_ids)].user)
        val = manifest[manifest["id"].isin(val_ids)]
        for (label, user, activity, room), group in val[val["label"].isin(["IRobot", "Fan"])].groupby(["label", "user", "activity", "environment"]):
            entry = {"split": k, "class": label, "device": user, "recording": activity, "room": room, "windows": int(len(group)), "same_device_in_training": user in train_users,
                     "training_windows_of_this_device": int(manifest[manifest["id"].isin(train_ids) & (manifest["user"] == user)].shape[0]), "models": {}}
            for model in models:
                z = np.load(MODELS_DIR / model / column / "probabilities.npz", allow_pickle=False)
                y, predicted = label_of.loc[z["ids_val"]].to_numpy(), z["proba_val"].argmax(axis=1)
                c, in_group = CLASS_INDEX[label], np.isin(z["ids_val"], group["id"].to_numpy())
                tp, fn, fp = int(((predicted == c) & (y == c)).sum()), int(((predicted != c) & (y == c)).sum()), int(((predicted == c) & (y != c)).sum())
                recall_group = float((predicted[in_group] == c).mean())
                precision = tp / max(tp + fp, 1)
                entry["models"][model] = {"recall_of_this_group": round(recall_group, 3), "precision_of_class_in_split": round(precision, 3),
                                          "f1_of_class_in_split": round(2 * tp / max(2 * tp + fp + fn, 1), 3),
                                          "called": {CLASSES[i]: int(n) for i, n in enumerate(np.bincount(predicted[in_group], minlength=N)) if n}}
            rows.append(entry)
    return rows


def run() -> dict:
    selection = json.loads((EVALUATION_DIR / "model_selection.json").read_text())
    main = selection["main_model"]
    for k in range(N_SPLITS):
        if fingerprint(main, k) != selection["candidates"][main]["test_probability_fingerprint"][f"split_{k}"]:
            raise SystemExit(f"{main} split_{k}: the test probabilities changed after the choice was recorded; stop")
    manifest, splits = pd.read_csv(PREP_DIR / "manifest.csv"), pd.read_csv(PREP_DIR / "splits.csv")
    meta_all = manifest.set_index("id")
    classes_in_room = manifest.groupby("environment")["label"].agg(lambda s: sorted(set(s))).to_dict()
    result = {"main_model": main, "classes_in_each_room_of_the_dataset": classes_in_room, "splits": {}}
    normalised = []
    for k in range(N_SPLITS):
        column = f"split_{k}"
        z = np.load(MODELS_DIR / main / column / "probabilities.npz", allow_pickle=False)
        ids, proba = z["ids_test"], z["proba_test"]
        y, predicted = splits.set_index("id").loc[ids, "label"].map(CLASS_INDEX).to_numpy(), proba.argmax(axis=1)
        meta = meta_all.loc[ids].reset_index()
        meta["group_id"] = splits.set_index("id").loc[ids, "group_id"].to_numpy()
        train_rooms = set(manifest[manifest["id"].isin(splits.loc[splits[column] == "train", "id"])].environment)
        matrix = confusion(y, predicted)
        normalised.append(row_normalised(matrix))
        result["splits"][column] = {"confusion_matrix": matrix.tolist(), "human_vs_rest": human_vs_rest(y, predicted), "confident_mistakes": confident_mistakes(proba, y),
                                    "rooms": room_table(y, predicted, meta, train_rooms), "room_types": room_type_table(y, predicted, meta, classes_in_room), "groups": group_table(y, predicted, meta)}
    mean = np.mean(normalised, axis=0)
    result["mean_row_normalised_confusion"] = np.round(mean, 3).tolist()
    result["key_confusions_mean"] = {f"{CLASSES[a]} called {CLASSES[b]}": round(float(mean[a, b]), 3) for a, b in [(0, 3), (3, 0), (1, 0), (0, 1), (0, 2), (2, 0), (1, 3), (3, 1), (2, 3), (3, 2), (2, 1), (1, 2)]}
    result["same_device_validation"] = same_device(manifest, splits, list(CANDIDATES))
    return result


def markdown(result: dict) -> str:
    L = [f"# Error analysis of {CANDIDATES[result['main_model']]} (test rows)\n", "## Confusion, mean of the three splits (rows = true class, share of its windows)\n",
         "| True \\ called | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * N]
    for i, c in enumerate(CLASSES):
        L.append(f"| {c} | " + " | ".join(f"{v:.2f}" for v in result["mean_row_normalised_confusion"][i]) + " |")
    L += ["", "Counts per split (rows = true class):", ""]
    for split, e in result["splits"].items():
        L.append(f"- {split}: " + "; ".join(f"{CLASSES[i]} {row}" for i, row in enumerate(e["confusion_matrix"])))
    L += ["", "## Key confusions (mean share of the true class)\n", "| Confusion | Share |", "|---|---|"] + [f"| {k} | {v:.3f} |" for k, v in result["key_confusions_mean"].items()]
    L += ["", "## Is it a person? (Human against everything else)\n", "| Split | Precision | Recall | F1 | Pet called Human | IRobot called Human | Fan called Human | Humans called Pet / IRobot / Fan |", "|---|---|---|---|---|---|---|---|"]
    for split, e in result["splits"].items():
        h = e["human_vs_rest"]
        L.append(f"| {split} | {h['precision']:.3f} | {h['recall']:.3f} | {h['f1']:.3f} | " + " | ".join(f"{h['false_alarms'][c]['share_of_that_class']:.1%} ({h['false_alarms'][c]['windows']})" for c in ("Pet", "IRobot", "Fan"))
                 + " | " + " / ".join(str(h["misses_called"][c]) for c in ("Pet", "IRobot", "Fan")) + " |")
    L += ["", "## Confident mistakes (highest probability at least 0.9)\n", "| Split | Windows that confident | Wrong among them | Share of all errors that are confident | Correct windows below 0.9 | Confident errors by true class |", "|---|---|---|---|---|---|"]
    for split, e in result["splits"].items():
        m = e["confident_mistakes"]
        L.append(f"| {split} | {m['windows_at_least_0.9']:.1%} | {m['wrong_among_those']:.1%} | {m['share_of_all_errors_that_are_confident']:.1%} | {m['share_of_correct_windows_below_0.9']:.1%} | {m['confident_errors_by_true_class']} |")
    L += ["", "## Rooms with the most errors (per split, top 4)\n", "| Split | Room | In training | Windows | Error rate | Errors | Classes |", "|---|---|---|---|---|---|---|"]
    for split, e in result["splits"].items():
        for r in e["rooms"][:4]:
            L.append(f"| {split} | {r['room']} | {r['in_training']} | {r['windows']} | {r['error_rate']:.1%} | {r['errors']} | {r['classes']} |")
    L += ["", "Rooms of the dataset and the classes recorded in them: " + "; ".join(f"{r} {v}" for r, v in result["classes_in_each_room_of_the_dataset"].items()), "",
          "| Split | Windows | Rooms with one class: windows / error rate | Rooms shared by several classes: windows / error rate |", "|---|---|---|---|"]
    for split, e in result["splits"].items():
        for scope, label in (("all_windows", "all"), ("human_and_pet_only", "Human and Pet only")):
            t = e["room_types"][scope]
            L.append(f"| {split} | {label} | {t['rooms_with_one_class']['windows']} / {t['rooms_with_one_class']['error_rate']} | {t['rooms_shared_by_several_classes']['windows']} / {t['rooms_shared_by_several_classes']['error_rate']} |")
    L += ["", "## Recording groups: the rare classes, and the three worst groups of the other classes\n", "| Split | Class | Device/user | Recording | Room | Windows | Accuracy | Mostly called | Share |", "|---|---|---|---|---|---|---|---|---|"]
    for split, e in result["splits"].items():
        groups = e["groups"]
        shown = [g for g in groups if g["class"] in ("IRobot", "Fan")] + sorted((g for g in groups if g["class"] in ("Human", "Pet")), key=lambda g: g["accuracy"])[:3]
        for g in shown:
            L.append(f"| {split} | {g['class']} | {g['user']} | {g['activity']} | {g['room']} | {g['windows']} | {g['accuracy']:.2f} | {g['most_called']} | {g['called_share']:.2f} |")
    L += ["", "## Same device in training or not (VALIDATION rows; rare-class groups; three candidates)\n",
          "| Split | Class | Device | Recording | Room | Windows | Same device in training (windows) | " + " | ".join(f"{CANDIDATES[m]}: recall / F1" for m in CANDIDATES) + " |", "|---|---|---|---|---|---|---|" + "---|" * len(CANDIDATES)]
    for r in result["same_device_validation"]:
        L.append(f"| split_{r['split']} | {r['class']} | {r['device']} | {r['recording']} | {r['room']} | {r['windows']} | {r['same_device_in_training']} ({r['training_windows_of_this_device']}) | "
                 + " | ".join(f"{r['models'][m]['recall_of_this_group']:.2f} / {r['models'][m]['f1_of_class_in_split']:.2f}" for m in CANDIDATES) + " |")
    L.append("\nF1 is the F1 of that class over the whole validation set of the split; recall is for this group only.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    out = run()
    (EVALUATION_DIR / "error_analysis.json").write_text(json.dumps(out, indent=1))
    (EVALUATION_DIR / "error_analysis.md").write_text(markdown(out))
    print(markdown(out))
