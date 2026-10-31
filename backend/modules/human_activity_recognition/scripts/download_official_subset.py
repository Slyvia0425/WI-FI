#!/usr/bin/env python3
"""Download a deterministic real-data subset for Module B from Kaggle.

The Kaggle dataset exposes 55k+ individual H5 files. This script downloads metadata, official train/val/test split
lists and a configurable number of real sessions per class per split. It never requests the full 81 GB archive.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess  # nosec B404
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DESTINATION = ROOT / "backend/modules/human_activity_recognition/local_data/csi_bench_module_b"
DATASET = "guozhenjennzhu/csi-bench"
REMOTE_META = {
    "sample_metadata.csv": "Multitask/HumanActivityRecognition/metadata/sample_metadata.csv",
    "label_mapping.json": "Multitask/HumanActivityRecognition/metadata/label_mapping.json",
    "device_mapping.json": "Multitask/HumanActivityRecognition/metadata/device_mapping.json",
}
SPLIT_NAMES = ("train", "val", "test")


def find_kaggle() -> Path:
    sibling = Path(sys.executable).with_name("kaggle")
    if sibling.exists():
        return sibling
    discovered = shutil.which("kaggle")
    if discovered:
        return Path(discovered)
    raise FileNotFoundError("kaggle CLI not found; install it with `python -m pip install kaggle`")


def run_kaggle(kaggle_bin: Path, remote_path: str, output: Path, *, attempts: int = 3) -> None:
    output.mkdir(parents=True, exist_ok=True)
    command = [
        str(kaggle_bin),
        "datasets",
        "download",
        "-d",
        DATASET,
        "-f",
        remote_path,
        "-p",
        str(output),
        "--force",
        "-q",
    ]
    last: subprocess.CalledProcessError | None = None
    for attempt in range(attempts):
        result = subprocess.run(command, capture_output=True, text=True, check=False)  # nosec B603  # noqa: S603
        if result.returncode == 0:
            return
        last = subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)
        time.sleep(0.5 * (2**attempt))
    raise last or RuntimeError("download failed")


def _round_robin(candidates: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    by_group: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in sorted(candidates, key=lambda item: (item["user"], item["device"], item["environment"], item["id"])):
        by_group[(row["user"], row["device"], row["environment"])].append(row)
    groups = sorted(by_group)
    selected: list[dict[str, str]] = []
    depth = 0
    while len(selected) < limit and groups:
        progressed = False
        for group in groups:
            if depth < len(by_group[group]):
                selected.append(by_group[group][depth])
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
        depth += 1
    return selected


def select_ids(
    rows: list[dict[str, str]], split_ids: dict[str, list[str]], per_class: int, num_sub: int | None = None
) -> dict[str, list[str]]:
    by_id = {row["id"]: row for row in rows}
    selected: dict[str, list[str]] = {}
    for split, ids in split_ids.items():
        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for sample_id in ids:
            row = by_id.get(sample_id)
            if row is not None and (num_sub is None or int(row["num_sub"]) == num_sub):
                groups[row["label"]].append(row)
        chosen: list[str] = []
        for label in sorted(groups):
            chosen.extend(row["id"] for row in _round_robin(groups[label], per_class))
        selected[split] = chosen
    return selected


def remote_h5_path(metadata_path: str) -> str:
    parts = PurePosixPath(metadata_path).parts
    if "sub_Human_h5" not in parts:
        raise ValueError(f"metadata path does not contain sub_Human_h5: {metadata_path}")
    tail = parts[parts.index("sub_Human_h5") + 1 :]
    return str(PurePosixPath("Multitask", "sub_Human_h5", *tail))


def download_one(kaggle_bin: Path, row: dict[str, str], destination: Path, staging: Path) -> tuple[str, int]:
    remote = remote_h5_path(row["file_path"])
    target = destination / remote
    if target.exists():
        return row["id"], target.stat().st_size
    with tempfile.TemporaryDirectory(dir=staging) as temp:
        temp_path = Path(temp)
        run_kaggle(kaggle_bin, remote, temp_path)
        downloaded = temp_path / PurePosixPath(remote).name
        if not downloaded.exists():
            raise FileNotFoundError(f"downloaded file missing: {downloaded}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(downloaded), target)
        return row["id"], target.stat().st_size


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download and materialize a deterministic Module B real-data subset")
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--samples-per-class", type=int, default=4, help="sessions per activity per official split")
    parser.add_argument(
        "--train-samples-per-class",
        type=int,
        default=None,
        help="optional train override; validation and test keep --samples-per-class",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--num-sub", type=int, default=232, help="filter sessions by subcarrier count; use 0 for all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--refresh-metadata", action="store_true")
    parser.add_argument("--prune", action="store_true", help="remove H5 files not referenced by the new subset")
    args = parser.parse_args(argv)
    if args.samples_per_class < 1:
        raise SystemExit("--samples-per-class must be positive")
    if args.train_samples_per_class is not None and args.train_samples_per_class < 1:
        raise SystemExit("--train-samples-per-class must be positive")
    kaggle_bin = find_kaggle()
    destination = args.destination.expanduser().resolve()
    staging = destination / "_staging"
    staging.mkdir(parents=True, exist_ok=True)
    for filename, remote in REMOTE_META.items():
        if args.refresh_metadata or not (staging / filename).exists():
            run_kaggle(kaggle_bin, remote, staging)
    for split in SPLIT_NAMES:
        local_split = staging / f"{split}_id.json"
        if args.refresh_metadata or not local_split.exists():
            run_kaggle(kaggle_bin, f"Multitask/HumanActivityRecognition/splits/{split}_id.json", staging)

    with (staging / "sample_metadata.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    split_ids = {split: json.loads((staging / f"{split}_id.json").read_text(encoding="utf-8")) for split in SPLIT_NAMES}
    num_sub = args.num_sub or None
    per_split_limits = {
        split: args.train_samples_per_class
        if split == "train" and args.train_samples_per_class
        else args.samples_per_class
        for split in SPLIT_NAMES
    }
    selected = {
        split: select_ids(rows, {split: split_ids[split]}, per_split_limits[split], num_sub)[split]
        for split in SPLIT_NAMES
    }
    selected_set = {sample_id for ids in selected.values() for sample_id in ids}
    selected_rows = [
        row for row in rows if row["id"] in selected_set and (num_sub is None or int(row["num_sub"]) == num_sub)
    ]
    if not selected_rows:
        raise SystemExit("no samples selected")
    manifest = {
        "dataset": DATASET,
        "source": "Kaggle official HumanActivityRecognition metadata/splits and selected H5 sessions",
        "license": "CC BY-NC-ND 4.0",
        "samples_per_class_per_split": per_split_limits,
        "num_sub": num_sub,
        "counts": {split: len(ids) for split, ids in selected.items()},
        "classes": sorted({row["label"] for row in selected_rows}),
        "files": len(selected_rows),
    }
    print(json.dumps(manifest, indent=2))
    if args.dry_run:
        return 0

    task = destination / "Multitask/HumanActivityRecognition"
    (task / "metadata").mkdir(parents=True, exist_ok=True)
    (task / "splits").mkdir(parents=True, exist_ok=True)
    shutil.copy2(staging / "label_mapping.json", task / "metadata/label_mapping.json")
    shutil.copy2(staging / "device_mapping.json", task / "metadata/device_mapping.json")
    with (task / "metadata/sample_metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or [])
        writer.writeheader()
        writer.writerows(selected_rows)
    for split, ids in selected.items():
        (task / f"splits/{split}_id.json").write_text(json.dumps(ids, indent=2) + "\n", encoding="utf-8")

    failures: list[dict[str, str]] = []
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, kaggle_bin, row, destination, staging): row for row in selected_rows}
        for future in as_completed(futures):
            row = futures[future]
            try:
                _, size = future.result()
                total_bytes += size
            except Exception as exc:  # noqa: BLE001 - report per-file failures
                failures.append({"id": row["id"], "error": str(exc)})
    manifest["downloaded_bytes"] = total_bytes
    manifest["failures"] = failures
    if args.prune and not failures:
        selected_paths = {(destination / remote_h5_path(row["file_path"])).resolve() for row in selected_rows}
        for file in (destination / "Multitask/sub_Human_h5").rglob("*.h5"):
            if file.resolve() not in selected_paths:
                file.unlink()
        for directory in sorted((destination / "Multitask/sub_Human_h5").rglob("*"), reverse=True):
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
        manifest["pruned_to_selected_files"] = True
    (destination / "subset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"ok": not failures, "downloaded_bytes": total_bytes, "failures": failures}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
