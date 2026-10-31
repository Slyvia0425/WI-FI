"""Training-only normalisation statistics, serialisable with the model checkpoint."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..errors import DataValidationError


@dataclass(frozen=True)
class FeatureStats:
    mean: list[float]
    std: list[float]
    mode: str = "global_zscore"

    @classmethod
    def fit(cls, samples: Iterable[np.ndarray], *, mode: str = "global_zscore") -> FeatureStats:
        if mode not in {"global_zscore", "window_zscore"}:
            raise DataValidationError("unknown normalisation mode", {"mode": mode})
        arrays = [np.asarray(sample, dtype=np.float32) for sample in samples]
        if not arrays:
            raise DataValidationError("cannot fit normalisation on no samples")
        width = arrays[0].shape[0]
        if any(array.ndim != 2 or array.shape[0] != width for array in arrays):
            raise DataValidationError("all training windows must share feature dimension")
        if mode == "window_zscore":
            # Each session has its own gain and phase baseline.  Normalising each
            # subcarrier's time series removes that nuisance variation while
            # preserving the temporal activity pattern.
            return cls(mean=[0.0] * width, std=[1.0] * width, mode=mode)
        stacked = np.concatenate(arrays, axis=1)
        mean = stacked.mean(axis=1)
        std = stacked.std(axis=1)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean=mean.tolist(), std=std.tolist(), mode=mode)

    def transform(self, array: np.ndarray) -> np.ndarray:
        values = np.asarray(array, dtype=np.float32)
        if values.ndim != 2:
            raise DataValidationError("normalisation expects a two-dimensional window")
        if self.mode == "window_zscore":
            center = values.mean(axis=1, keepdims=True)
            scale = values.std(axis=1, keepdims=True)
            scale = np.where(scale < 1e-8, 1.0, scale)
            return (values - center) / scale
        mean: np.ndarray = np.asarray(self.mean, dtype=np.float32).reshape(-1, 1)
        std: np.ndarray = np.asarray(self.std, dtype=np.float32).reshape(-1, 1)
        if values.shape[0] != mean.shape[0]:
            raise DataValidationError("feature dimension does not match normalisation statistics")
        return (values - mean) / std

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeatureStats:
        mode = str(payload.get("mode", "global_zscore"))
        if mode not in {"global_zscore", "window_zscore"}:
            raise DataValidationError("unknown normalisation mode", {"mode": mode})
        return cls(
            mean=[float(v) for v in payload["mean"]],
            std=[float(v) for v in payload["std"]],
            mode=mode,
        )
