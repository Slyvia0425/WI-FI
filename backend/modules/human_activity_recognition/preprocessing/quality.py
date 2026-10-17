"""Input quality checks. Invalid data never reaches the model; degraded data is flagged."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import DataValidationError


@dataclass(frozen=True)
class QualityReport:
    flag: str
    score: float
    codes: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.flag != "invalid"


def orient_window(array: np.ndarray, expected_feature_size: int) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2:
        raise DataValidationError("CSI window must be two-dimensional", {"shape": list(values.shape)})
    if values.shape[0] == expected_feature_size:
        return values
    if values.shape[1] == expected_feature_size:
        return values.T
    raise DataValidationError(
        "CSI window does not contain the configured feature dimension",
        {"shape": list(values.shape), "expected_feature_size": expected_feature_size},
    )


def check_quality(
    array: np.ndarray,
    *,
    expected_feature_size: int,
    min_frames: int,
    max_abs_amplitude: float,
) -> QualityReport:
    values = np.asarray(array)
    if values.size == 0:
        return QualityReport("invalid", 0.0, ("empty",))
    if not np.isfinite(values).all():
        return QualityReport("invalid", 0.0, ("non_finite",))
    oriented = orient_window(values, expected_feature_size)
    codes: list[str] = []
    if oriented.shape[1] < min_frames:
        codes.append("too_short")
    if float(np.max(np.abs(oriented))) > max_abs_amplitude:
        codes.append("amplitude_out_of_range")
    if float(np.std(oriented)) < 1e-9:
        codes.append("constant_signal")
    if "too_short" in codes or "amplitude_out_of_range" in codes:
        return QualityReport("invalid", 0.0, tuple(codes))
    score = max(0.0, min(1.0, 1.0 - 0.25 * len(codes)))
    return QualityReport("degraded" if codes else "good", score, tuple(codes))


def require_valid(report: QualityReport, *, context: str) -> None:
    if not report.ok:
        raise DataValidationError("CSI window failed quality checks", {"context": context, "codes": list(report.codes)})
