"""Run a small, read-only audit of the CSI-Bench breathing dataset.

This deliberately samples the dataset instead of loading all 100,000 HDF5
files. It is intended to establish the real storage schema before a loader or
model is implemented.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/BreathingDetection"),
        help="BreathingDetection dataset directory",
    )
    parser.add_argument("--samples-per-group", type=int, default=1)
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print aggregate HDF5 signatures instead of every sampled record",
    )
    return parser.parse_args()


def scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def inspect_h5(path: Path) -> dict[str, Any]:
    datasets: list[dict[str, Any]] = []
    with h5py.File(path, "r") as handle:
        root_attrs = {key: scalar(value) for key, value in handle.attrs.items()}

        def visit(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            values = obj[...]
            numeric = np.issubdtype(values.dtype, np.number)
            finite = np.isfinite(values) if numeric else None
            datasets.append(
                {
                    "name": name,
                    "shape": list(values.shape),
                    "dtype": str(values.dtype),
                    "attrs": {key: scalar(value) for key, value in obj.attrs.items()},
                    "nan_count": int(np.isnan(values).sum()) if numeric else None,
                    "inf_count": int(np.isinf(values).sum()) if numeric else None,
                    "min": float(values[finite].min()) if numeric and finite.any() else None,
                    "max": float(values[finite].max()) if numeric and finite.any() else None,
                    "mean": float(values[finite].mean()) if numeric and finite.any() else None,
                    "std": float(values[finite].std()) if numeric and finite.any() else None,
                }
            )

        handle.visititems(visit)
    return {"root_attrs": root_attrs, "datasets": datasets}


def main() -> None:
    args = parse_args()
    root = args.data_root.resolve()
    metadata_path = root / "metadata" / "sample_metadata.csv"
    with metadata_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    # One deterministic sample per label/user/difficulty/device combination.
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row["label"], row["user"], row["difficulty"], row["device"])
        groups[key].append(row)

    selected: list[dict[str, str]] = []
    for key in sorted(groups):
        selected.extend(groups[key][: args.samples_per_group])

    records = []
    for row in selected:
        path = root / row["file_path"].removeprefix("./")
        records.append(
            {
                "sample_id": row["id"],
                "label": row["label"],
                "user": row["user"],
                "difficulty": row["difficulty"],
                "device": row["device"],
                "environment": row["environment"],
                "deployment": row["deplyoment"],
                "relative_path": str(path.relative_to(root)),
                "h5": inspect_h5(path),
            }
        )

    split_counts = {}
    for split_path in sorted((root / "splits").glob("*.json")):
        split_counts[split_path.stem] = len(json.loads(split_path.read_text()))

    result = {
        "data_root": str(root),
        "metadata_rows": len(rows),
        "label_counts": dict(Counter(row["label"] for row in rows)),
        "split_counts": split_counts,
        "sampling_strategy": "first sample per label/user/difficulty/device group",
        "sampled_files": len(records),
        "records": records,
    }
    if args.summary:
        signatures: Counter[tuple[str, tuple[int, ...], str]] = Counter()
        nan_count = 0
        inf_count = 0
        for record in records:
            for dataset in record["h5"]["datasets"]:
                signatures[
                    (dataset["name"], tuple(dataset["shape"]), dataset["dtype"])
                ] += 1
                nan_count += dataset["nan_count"] or 0
                inf_count += dataset["inf_count"] or 0
        result["dataset_signatures"] = [
            {"name": key[0], "shape": list(key[1]), "dtype": key[2], "count": count}
            for key, count in signatures.most_common()
        ]
        result["sampled_nan_count"] = nan_count
        result["sampled_inf_count"] = inf_count
        result["record_examples"] = records[:5]
        del result["records"]
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
