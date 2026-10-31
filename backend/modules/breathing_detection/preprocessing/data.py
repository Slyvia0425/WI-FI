"""Read-only dataset and preprocessing for CSI-Bench BreathingDetection."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np


PreprocessingMode = Literal["project_pipeline", "official_reproduction"]

LABEL_TO_INDEX = {"empty": 0, "sleep": 1}
PROJECT_TIME_STEPS = 300
OFFICIAL_TIME_STEPS = 500
OFFICIAL_FEATURES = 232


class DataValidationError(ValueError):
    """Raised when a sample cannot be interpreted without guessing."""


@dataclass(frozen=True)
class BreathingSample:
    x: np.ndarray
    label: int
    sample_id: str
    split: str
    assessable: bool
    quality_flags: tuple[str, ...]
    metadata: dict[str, Any]


def _optional_scenario_fields(deployment: str) -> tuple[bool | None, bool | None]:
    """Return (fan_on, los) only when deployment text states them explicitly."""
    compact = deployment.lower().replace("_", "").replace("-", "")
    fan_on = True if "fanon" in compact else False if "fanoff" in compact else None
    if "nlos" in compact:
        los = False
    elif "los" in compact:
        los = True
    else:
        los = None
    return fan_on, los


def _zscore(values: np.ndarray, epsilon: float) -> np.ndarray:
    mean = float(values.mean(dtype=np.float64))
    std = float(values.std(dtype=np.float64))
    if std < epsilon:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - mean) / max(std, epsilon)).astype(np.float32, copy=False)


class BreathingDataset:
    """Lazy, read-only view of one official CSI-Bench split.

    ``project_pipeline`` returns ``[C, K, T]`` and preserves the real number of
    subcarriers and the 300-point window. ``official_reproduction`` implements
    the authors' generic benchmark preprocessing and returns ``[1, 500, 232]``.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        *,
        mode: PreprocessingMode = "project_pipeline",
        normalize: bool = True,
        epsilon: float = 1e-8,
        flat_std_threshold: float = 1e-6,
    ) -> None:
        if mode not in ("project_pipeline", "official_reproduction"):
            raise ValueError(f"Unsupported preprocessing mode: {mode}")
        self.root = Path(data_root).expanduser().resolve()
        self.split = split
        self.mode = mode
        self.normalize = normalize
        self.epsilon = epsilon
        self.flat_std_threshold = flat_std_threshold

        metadata_path = self.root / "metadata" / "sample_metadata.csv"
        split_path = self.root / "splits" / f"{split}.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        if not split_path.is_file():
            raise FileNotFoundError(f"Split file not found: {split_path}")

        with metadata_path.open(newline="", encoding="utf-8") as stream:
            metadata_by_id = {row["id"]: row for row in csv.DictReader(stream)}
        sample_ids = json.loads(split_path.read_text(encoding="utf-8"))
        if not isinstance(sample_ids, list) or not all(isinstance(x, str) for x in sample_ids):
            raise DataValidationError(f"Split must contain a JSON list of IDs: {split_path}")
        missing = [sample_id for sample_id in sample_ids if sample_id not in metadata_by_id]
        if missing:
            raise DataValidationError(
                f"{len(missing)} split IDs are absent from metadata; first: {missing[0]}"
            )
        self._rows = [metadata_by_id[sample_id] for sample_id in sample_ids]

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> BreathingSample:
        row = self._rows[index]
        label_name = row["label"]
        if label_name not in LABEL_TO_INDEX:
            raise DataValidationError(
                f"Unknown breathing label {label_name!r} for sample {row['id']}"
            )

        relative_path = row["file_path"].removeprefix("./")
        path = self.root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"CSI file not found for {row['id']}: {path}")
        with h5py.File(path, "r") as handle:
            if "CSI_amps" not in handle:
                raise DataValidationError(
                    f"CSI_amps is missing in {path}; available keys: {list(handle.keys())}"
                )
            raw = np.asarray(handle["CSI_amps"], dtype=np.float32)

        if raw.ndim != 3 or raw.shape[2] != 1:
            raise DataValidationError(
                f"Expected CSI_amps layout [K,T,1], got {raw.shape} for {row['id']}"
            )
        if raw.shape[1] != PROJECT_TIME_STEPS:
            raise DataValidationError(
                f"Expected {PROJECT_TIME_STEPS} time points, got {raw.shape[1]} for {row['id']}"
            )
        if raw.shape[0] < 1:
            raise DataValidationError(f"Sample has no subcarriers: {row['id']}")

        flags: list[str] = []
        finite = np.isfinite(raw)
        if not finite.all():
            flags.append("NON_FINITE_VALUES")
            raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        if float(raw.std(dtype=np.float64)) < self.flat_std_threshold:
            flags.append("FLAT_SIGNAL")

        if self.normalize:
            raw = _zscore(raw, self.epsilon)

        if self.mode == "project_pipeline":
            # Stored [K,T,C] -> canonical [C,K,T].
            values = np.transpose(raw, (2, 0, 1))
        else:
            # Match the authors' benchmark loader: [K,T,C] -> [C,T,K], then
            # copy into the upper-left of a fixed [1,500,232] zero tensor.
            time_first = np.transpose(raw, (2, 1, 0))
            values = np.zeros(
                (1, OFFICIAL_TIME_STEPS, OFFICIAL_FEATURES), dtype=np.float32
            )
            copy_time = min(time_first.shape[1], OFFICIAL_TIME_STEPS)
            copy_features = min(time_first.shape[2], OFFICIAL_FEATURES)
            values[:, :copy_time, :copy_features] = time_first[
                :, :copy_time, :copy_features
            ]

        deployment = row.get("deplyoment") or row.get("deployment") or ""
        fan_on, los = _optional_scenario_fields(deployment)
        metadata = {
            "file_path": relative_path,
            "label_name": label_name,
            "participant_id": row.get("user") or None,
            "environment_id": row.get("environment") or None,
            "device_id": row.get("device") or None,
            "session_id": row.get("session") or None,
            "difficulty": row.get("difficulty") or None,
            "deployment": deployment or None,
            "fan_on": fan_on,
            "los": los,
            "sampling_rate_hz": 30.0,
            "window_seconds": PROJECT_TIME_STEPS / 30.0,
            "original_shape": list(raw.shape),
        }
        return BreathingSample(
            x=np.ascontiguousarray(values),
            label=LABEL_TO_INDEX[label_name],
            sample_id=row["id"],
            split=self.split,
            assessable=not flags,
            quality_flags=tuple(flags),
            metadata=metadata,
        )
