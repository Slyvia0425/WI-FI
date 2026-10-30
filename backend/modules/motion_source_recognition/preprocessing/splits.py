"""Step 2a of preprocessing: three train / validation / test splits made by recording group.

Run from the repository root (needs outputs/prep/manifest.csv and quality.csv):
    python3 -m backend.modules.motion_source_recognition.preprocessing.splits

Rules
- The unit is the recording group (user, activity, environment), see manifest.py. A group is never in two of train, validation and test.
- Classes with at least ROTATE_BELOW groups (Human, Pet): groups are shuffled with the split's seed and filled into test, then validation,
  until each holds about 15% of the class's samples.
- Classes with fewer groups (Fan with 5, IRobot with 3): groups rotate, so every split tests a different group. The validation group is
  the next group in order when the class has at least CYCLIC_VAL_FROM groups (Fan); with fewer (IRobot) it is the smaller of the two
  groups left, so the larger one trains. Each IRobot group therefore serves once as test, and every split is strictly group-separated.
- Samples flagged bad in quality.csv would be left out; the data has none.
- Some windows exist twice with identical content (614 pairs, always inside one class). When the two copies end up in different subsets,
  the copy in the lower-priority subset is marked "duplicate" and not used anywhere: priority is train, then test, then validation. So no
  test window has an exact copy in training, and no validation window has one in training or test.

Writes outputs/prep/splits.csv, splits_counts.csv and splits_report.json.
"""
import json

import pandas as pd

from .config import CLASSES, PREP_DIR, SPLITS

N_SPLITS = SPLITS["n_splits"]
TEST_FRACTION = SPLITS["test_fraction"]
VAL_FRACTION = SPLITS["val_fraction"]
ROTATE_BELOW = SPLITS["rotate_below"]   # classes with fewer recording groups than this rotate groups instead of shuffling them
CYCLIC_VAL_FROM = SPLITS["cyclic_val_from"]  # rotating classes with at least this many groups take the next group as validation; fewer take the smaller one left
SPLIT_COLUMNS = [f"split_{k}" for k in range(N_SPLITS)]
SUBSETS = ["train", "val", "test", "duplicate"]
PRIORITY = SPLITS["priority"]  # which copy of an identical window survives when copies fall into different subsets


def _shuffled_assignment(groups: pd.DataFrame, seed: int) -> dict:
    order = groups.sample(frac=1, random_state=seed).index
    total = groups["n"].sum()
    out, test, val = {}, 0, 0
    for gid in order:
        n = groups.at[gid, "n"]
        if test < TEST_FRACTION * total:
            out[gid], test = "test", test + n
        elif val < VAL_FRACTION * total:
            out[gid], val = "val", val + n
        else:
            out[gid] = "train"
    return out


def _rotating_assignment(groups: pd.DataFrame, k: int) -> dict:
    ids = sorted(groups.index)
    test = ids[k % len(ids)]
    out = {g: "train" for g in ids}
    out[test] = "test"
    if len(ids) >= CYCLIC_VAL_FROM:
        out[ids[(k + 1) % len(ids)]] = "val"
    else:
        left = groups.drop(index=test)
        out[left["n"].idxmin()] = "val"
    return out


def remove_duplicates(splits: pd.DataFrame, column: str) -> int:
    """Mark copies of identical windows that sit in a lower-priority subset than another copy. Returns how many were marked."""
    rank = splits[column].map({s: i for i, s in enumerate(PRIORITY)})
    best = rank.groupby(splits["content_hash"]).transform("min")
    lose = rank > best
    splits.loc[lose, column] = "duplicate"
    return int(lose.sum())


def build_splits(manifest: pd.DataFrame) -> pd.DataFrame:
    out = manifest[["id", "group_id", "label", "user", "environment", "session", "content_hash"]].copy()
    for k, column in enumerate(SPLIT_COLUMNS):
        out[column] = "train"
        for label, rows in manifest.groupby("label"):
            groups = rows.groupby("group_id").size().rename("n").to_frame()
            if len(groups) >= ROTATE_BELOW:
                assignment = _shuffled_assignment(groups, seed=k)
            else:
                assignment = _rotating_assignment(groups, k)
            out.loc[rows.index, column] = rows["group_id"].map(assignment)
        remove_duplicates(out, column)
    return out


def leakage_checks(splits: pd.DataFrame) -> dict:
    """Group-level and content-level checks, per split. Every count must be 0."""
    report = {}
    for column in SPLIT_COLUMNS:
        used = splits[splits[column] != "duplicate"]
        hashes = {s: set(used.loc[used[column] == s, "content_hash"]) for s in ["train", "val", "test"]}
        groups = {s: set(splits.loc[splits[column] == s, "group_id"]) for s in ["train", "val", "test"]}
        by_class = {}
        for label in CLASSES:
            sub = splits[splits.label == label]
            g = {s: set(sub.loc[sub[column] == s, "group_id"]) for s in ["train", "val", "test"]}
            by_class[label] = {
                "test_groups_also_in_train": len(g["test"] & g["train"]),
                "test_groups_also_in_val": len(g["test"] & g["val"]),
                "val_groups_also_in_train": len(g["val"] & g["train"]),
            }
        report[column] = {
            "groups_in_test_and_train": len(groups["test"] & groups["train"]),
            "groups_in_test_and_val": len(groups["test"] & groups["val"]),
            "identical_windows_in_test_and_train": len(hashes["test"] & hashes["train"]),
            "identical_windows_in_val_and_train": len(hashes["val"] & hashes["train"]),
            "identical_windows_in_val_and_test": len(hashes["val"] & hashes["test"]),
            "windows_marked_duplicate": int((splits[column] == "duplicate").sum()),
            "by_class": by_class,
        }
    return report


def describe(splits: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows, extra = [], {}
    for column in SPLIT_COLUMNS:
        extra[column] = {}
        for label in CLASSES:
            sub = splits[splits.label == label]
            test = sub[sub[column] == "test"]
            train = sub[sub[column] == "train"]
            extra[column][label] = {
                "test_environments": sorted(test.environment.unique()),
                "test_environments_not_in_train": sorted(set(test.environment) - set(train.environment)),
                "test_users_also_in_train": f"{len(set(test.user) & set(train.user))} of {test.user.nunique()}",
            }
            for subset in SUBSETS:
                part = sub[sub[column] == subset]
                rows.append({"split": column, "subset": subset, "label": label, "groups": part.group_id.nunique(), "samples": len(part)})
    return pd.DataFrame(rows), extra


def run() -> pd.DataFrame:
    manifest = pd.read_csv(PREP_DIR / "manifest.csv")
    quality = pd.read_csv(PREP_DIR / "quality.csv")
    manifest = manifest.merge(quality[["id", "bad", "content_hash"]], on="id")
    manifest = manifest[~manifest["bad"]].reset_index(drop=True)

    splits = build_splits(manifest)
    counts, extra = describe(splits)
    checks = leakage_checks(splits)

    splits.drop(columns=["user", "environment", "session", "content_hash"]).to_csv(PREP_DIR / "splits.csv", index=False)
    counts.to_csv(PREP_DIR / "splits_counts.csv", index=False)
    report = {
        "parameters": {"test_fraction": TEST_FRACTION, "val_fraction": VAL_FRACTION, "rotate_below_groups": ROTATE_BELOW,
                       "cyclic_validation_from_groups": CYCLIC_VAL_FROM, "splits": N_SPLITS},
        "leakage_checks": checks,
        "per_class_details": extra,
    }
    (PREP_DIR / "splits_report.json").write_text(json.dumps(report, indent=1))

    table = counts.pivot_table(index=["split", "label"], columns="subset", values="samples", aggfunc="sum")[["train", "val", "test", "duplicate"]]
    print(table.to_string())
    print("\nGroups in test and also in train, per split:", {c: v["groups_in_test_and_train"] for c, v in checks.items()})
    print("Identical windows across subsets after removal:", {c: v["identical_windows_in_test_and_train"] + v["identical_windows_in_val_and_train"] + v["identical_windows_in_val_and_test"] for c, v in checks.items()})
    print("Windows marked duplicate:", {c: v["windows_marked_duplicate"] for c, v in checks.items()})
    return splits


if __name__ == "__main__":
    run()
