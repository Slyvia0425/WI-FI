"""Timestamp-aware, read-only adapter for the WiRe-Breath MATLAB release."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection

import numpy as np
from scipy.io import loadmat

from .data import DataValidationError, _zscore


WIRE_SUBCARRIERS = 234
WIRE_SOURCE_POINTS = 1500
WIRE_TARGET_RATE_HZ = 30.0
WIRE_WINDOW_SECONDS = 10.0


@dataclass(frozen=True)
class WireBreathSample:
    """One aligned window; this dataset has no empty/breathing class label."""

    x: np.ndarray
    respiratory_reference: np.ndarray
    sample_id: str
    recording_id: str
    assessable: bool
    quality_flags: tuple[str, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _WindowReference:
    path: Path
    start_seconds: float
    sample_id: str
    recording_id: str
    metadata: dict[str, Any]


def _mat_files(root: Path) -> list[Path]:
    """Find MAT files in normal folders or a flat Windows-style extraction."""
    return sorted(
        root.rglob("*.mat"),
        key=lambda path: path.relative_to(root).as_posix().replace("\\", "/"),
    )


def _metadata_value(metadata: Any, name: str) -> Any:
    if not hasattr(metadata, name):
        raise DataValidationError(f"WiRe-Breath metadata is missing {name!r}")
    value = getattr(metadata, name)
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        if value.size != 1:
            raise DataValidationError(f"Metadata field {name!r} is not scalar")
        value = value.item()
    return value


def _metadata_dict(metadata: Any) -> dict[str, Any]:
    session = _metadata_value(metadata, "session_id")
    session_id = None if session is None or str(session) in ("", "[]") else str(session)
    return {
        "file_name": str(_metadata_value(metadata, "file_name")),
        "room": str(_metadata_value(metadata, "room")),
        "participant_id": str(_metadata_value(metadata, "subject_id")),
        "posture": str(_metadata_value(metadata, "posture")),
        "environment": str(_metadata_value(metadata, "environment")),
        "session_id": session_id,
        "trial_id": int(_metadata_value(metadata, "trial_id")),
        "segment_id": int(_metadata_value(metadata, "segment_id")),
    }


def _recording_id(metadata: dict[str, Any]) -> str:
    session = metadata["session_id"] or "no-session"
    return "/".join(
        (
            metadata["room"],
            f"Subject{metadata['participant_id']}",
            metadata["posture"],
            metadata["environment"],
            session,
            f"Trial{metadata['trial_id']:02d}",
        )
    )


def _strict_time(name: str, values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(result) < 2 or not np.isfinite(result).all():
        raise DataValidationError(f"{name} must contain finite timestamps")
    if not np.all(np.diff(result) > 0):
        raise DataValidationError(f"{name} must be strictly increasing")
    return result


class WireBreathDataset:
    """Lazy windows from WiRe-Breath, aligned using the supplied timestamps.

    Windows from the same ``recording_id`` must always remain in one split.
    ``recording_ids`` enables a caller to apply a precomputed grouped split.
    """

    def __init__(
        self,
        data_root: str | Path,
        *,
        target_rate_hz: float = WIRE_TARGET_RATE_HZ,
        window_seconds: float = WIRE_WINDOW_SECONDS,
        stride_seconds: float = WIRE_WINDOW_SECONDS,
        max_gap_seconds: float = 0.1,
        normalize: bool = True,
        epsilon: float = 1e-8,
        flat_std_threshold: float = 1e-6,
        environments: Collection[str] | None = None,
        recording_ids: Collection[str] | None = None,
    ) -> None:
        if target_rate_hz <= 0 or window_seconds <= 0 or stride_seconds <= 0:
            raise ValueError("Rates, window length, and stride must be positive")
        self.root = Path(data_root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"WiRe-Breath root not found: {self.root}")
        self.target_rate_hz = float(target_rate_hz)
        self.window_seconds = float(window_seconds)
        self.stride_seconds = float(stride_seconds)
        self.max_gap_seconds = float(max_gap_seconds)
        self.normalize = normalize
        self.epsilon = epsilon
        self.flat_std_threshold = flat_std_threshold
        self.window_points = int(round(window_seconds * target_rate_hz))
        if not np.isclose(self.window_points / target_rate_hz, window_seconds):
            raise ValueError("window_seconds * target_rate_hz must be an integer")

        environment_filter = set(environments) if environments is not None else None
        recording_filter = set(recording_ids) if recording_ids is not None else None
        self._windows: list[_WindowReference] = []
        files = _mat_files(self.root)
        if not files:
            raise FileNotFoundError(f"No MAT files found under {self.root}")

        for path in files:
            payload = loadmat(
                path,
                squeeze_me=True,
                struct_as_record=False,
                variable_names=("time", "gt_time", "metadata"),
            )
            try:
                time = _strict_time("time", payload["time"])
                gt_time = _strict_time("gt_time", payload["gt_time"])
                metadata = _metadata_dict(payload["metadata"])
            except KeyError as error:
                raise DataValidationError(f"Missing {error.args[0]!r} in {path}") from error
            recording_id = _recording_id(metadata)
            if environment_filter is not None and metadata["environment"] not in environment_filter:
                continue
            if recording_filter is not None and recording_id not in recording_filter:
                continue

            common_start = max(float(time[0]), float(gt_time[0]))
            common_end = min(float(time[-1]), float(gt_time[-1]))
            count = int(np.floor((common_end - common_start - window_seconds) / stride_seconds)) + 1
            for window_index in range(max(0, count)):
                start = common_start + window_index * stride_seconds
                stem = metadata["file_name"].removesuffix(".mat")
                self._windows.append(
                    _WindowReference(
                        path=path,
                        start_seconds=start,
                        sample_id=f"{stem}@{start:.3f}s",
                        recording_id=recording_id,
                        metadata=metadata,
                    )
                )

        if not self._windows:
            raise DataValidationError("No complete aligned windows match the requested filters")

    @property
    def recording_ids(self) -> tuple[str, ...]:
        return tuple(sorted({window.recording_id for window in self._windows}))

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> WireBreathSample:
        reference = self._windows[index]
        payload = loadmat(
            reference.path,
            squeeze_me=True,
            struct_as_record=False,
            variable_names=("csi_abs", "time", "gt_signal", "gt_time"),
        )
        try:
            csi = np.asarray(payload["csi_abs"], dtype=np.float32)
            time = _strict_time("time", payload["time"])
            gt_signal = np.asarray(payload["gt_signal"], dtype=np.float32).reshape(-1)
            gt_time = _strict_time("gt_time", payload["gt_time"])
        except KeyError as error:
            raise DataValidationError(
                f"Missing {error.args[0]!r} in {reference.path}"
            ) from error
        if csi.shape != (WIRE_SOURCE_POINTS, WIRE_SUBCARRIERS):
            raise DataValidationError(
                f"Expected CSI shape {(WIRE_SOURCE_POINTS, WIRE_SUBCARRIERS)}, got {csi.shape}"
            )
        if len(gt_signal) != len(gt_time):
            raise DataValidationError("Ground-truth signal/time length mismatch")
        if not np.isfinite(csi).all() or not np.isfinite(gt_signal).all():
            raise DataValidationError("WiRe-Breath arrays contain non-finite values")

        start = reference.start_seconds
        target_time = start + np.arange(self.window_points, dtype=np.float64) / self.target_rate_hz
        csi_window = np.empty((self.window_points, WIRE_SUBCARRIERS), dtype=np.float32)
        for subcarrier in range(WIRE_SUBCARRIERS):
            csi_window[:, subcarrier] = np.interp(
                target_time, time, csi[:, subcarrier]
            )
        respiratory_reference = np.interp(target_time, gt_time, gt_signal).astype(np.float32)

        flags: list[str] = []
        covered = (time[:-1] <= target_time[-1]) & (time[1:] >= target_time[0])
        if np.any(np.diff(time)[covered] > self.max_gap_seconds):
            flags.append("LARGE_TIMESTAMP_GAP")
        if float(csi_window.std(dtype=np.float64)) < self.flat_std_threshold:
            flags.append("FLAT_SIGNAL")
        if self.normalize:
            csi_window = _zscore(csi_window, self.epsilon)
            respiratory_reference = _zscore(respiratory_reference, self.epsilon)

        metadata = dict(reference.metadata)
        metadata.update(
            {
                "source_path": str(reference.path.relative_to(self.root)),
                "window_start_seconds": start,
                "window_seconds": self.window_seconds,
                "sampling_rate_hz": self.target_rate_hz,
                "source_subcarriers": WIRE_SUBCARRIERS,
                "task_label": "breathing_present",
            }
        )
        return WireBreathSample(
            x=np.ascontiguousarray(csi_window.T[None, :, :]),
            respiratory_reference=np.ascontiguousarray(respiratory_reference),
            sample_id=reference.sample_id,
            recording_id=reference.recording_id,
            assessable=not flags,
            quality_flags=tuple(flags),
            metadata=metadata,
        )
