"""Step 1a of preprocessing: check the metadata table against the files and assign recording groups.

Run from the repository root:
    CSI_DATA_ROOT=/path/to/MotionSourceRecognition python3 -m backend.modules.motion_source_recognition.preprocessing.manifest

Writes outputs/prep/manifest.csv and outputs/prep/manifest_report.json.
Nothing here opens an h5 file; the content checks are in quality.py.
"""
import json
import os

import pandas as pd

from .config import PREP_DIR, data_root

# Only spellings that are clearly the same activity are merged. Used for description and analysis only;
# recording groups below always use the original names. Unclear names stay as they are.
ACTIVITY_ALIASES = {
    "walk": "Walking",
    "sneak": "Sneaking",
    "sneak2": "Sneaking",
    "run": "Running",
    "Laser_Pointer_`": "Laser_Pointer_1",  # backtick is a typo; Laser_Pointer_2 also exists
}


def normalise_activity(name: str) -> str:
    if name in ACTIVITY_ALIASES:
        return ACTIVITY_ALIASES[name]
    lowered = name.lower()
    if lowered.startswith("test") and name != "Test":  # test1 -> Test_1
        digits = "".join(c for c in lowered if c.isdigit())
        return f"Test_{digits}" if digits else name
    return name


def _official_splits(root) -> dict:
    out = {}
    for name in ["train_id", "val_id", "test_id"]:
        path = root / "splits" / f"{name}.json"
        if path.exists():
            out[name] = set(json.loads(path.read_text()))
    return out


def build_manifest() -> pd.DataFrame:
    root = data_root()
    df = pd.read_csv(root / "metadata" / "sample_metadata.csv")
    df["file_path"] = df["file_path"].str.removeprefix("./")
    # The metadata also has the dataset's easy / medium / hard label. It is not used: the paper only describes it in words
    # (signal quality, environment, subject complexity) without a rule, so it is left out to keep every sample on equal footing.
    df = df.drop(columns=["difficulty"], errors="ignore")

    on_disk = {
        os.path.relpath(os.path.join(dirpath, f), root)
        for dirpath, _, files in os.walk(root)
        for f in files
        if f.endswith(".h5")
    }
    listed = set(df["file_path"])

    df["file_exists"] = df["file_path"].isin(on_disk)
    df["activity_norm"] = df["activity"].map(normalise_activity)
    df["group_id"] = df.groupby(["user", "activity", "environment"], sort=True).ngroup()

    splits = _official_splits(root)
    split_of = {}
    for name in ["train_id", "val_id", "test_id"]:
        for sample_id in splits.get(name, ()):
            split_of[sample_id] = name.removesuffix("_id")
    df["official_split"] = df["id"].map(split_of).fillna("none")

    group_split = df[df.official_split != "none"].groupby("group_id")["official_split"].nunique()
    test_groups = set(df.loc[df.official_split == "test", "group_id"])
    train_groups = set(df.loc[df.official_split == "train", "group_id"])
    report = {
        "rows": len(df),
        "h5_files_on_disk": len(on_disk),
        "rows_without_file": int((~df["file_exists"]).sum()),
        "files_not_in_table": len(on_disk - listed),
        "duplicate_ids": int(df["id"].duplicated().sum()),
        "duplicate_paths": int(df["file_path"].duplicated().sum()),
        "recording_groups": int(df["group_id"].nunique()),
        "groups_per_class": df.groupby("label")["group_id"].nunique().to_dict(),
        "samples_per_class": df["label"].value_counts().to_dict(),
        "users_per_class": df.groupby("label")["user"].nunique().to_dict(),
        "environments_per_class": df.groupby("label")["environment"].unique().map(sorted).to_dict(),
        "activity_names_raw": int(df["activity"].nunique()),
        "activity_names_after_merge": int(df["activity_norm"].nunique()),
        "official_split_groups_in_more_than_one_split": int((group_split > 1).sum()),
        "official_test_groups_also_in_train": f"{len(test_groups & train_groups)} of {len(test_groups)}",
    }

    PREP_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PREP_DIR / "manifest.csv", index=False)
    (PREP_DIR / "manifest_report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))
    return df


if __name__ == "__main__":
    build_manifest()
