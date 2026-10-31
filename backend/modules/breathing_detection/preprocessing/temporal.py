"""Device-agnostic temporal representation for variable-subcarrier CSI."""

from __future__ import annotations

import numpy as np


TEMPORAL_FEATURE_NAMES = (
    "mean",
    "std",
    "q10",
    "q25",
    "median",
    "q75",
    "q90",
    "rms",
    "mean_absolute",
    "range_q90_q10",
)

TOP_VARIANCE_FEATURES = 32
INTERPOLATED_FEATURES = 232


def extract_temporal_statistics(x: np.ndarray) -> np.ndarray:
    """Convert project-layout ``[C,K,T]`` into fixed ``[T,10]`` statistics."""
    values = np.asarray(x, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"Expected [C,K,T], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Temporal statistics require finite input")
    signals = values.reshape(-1, values.shape[-1])
    quantiles = np.quantile(signals, [0.1, 0.25, 0.5, 0.75, 0.9], axis=0)
    result = np.stack(
        [
            signals.mean(axis=0),
            signals.std(axis=0),
            quantiles[0],
            quantiles[1],
            quantiles[2],
            quantiles[3],
            quantiles[4],
            np.sqrt(np.mean(signals**2, axis=0)),
            np.mean(np.abs(signals), axis=0),
            quantiles[4] - quantiles[0],
        ],
        axis=1,
    ).astype(np.float32)
    if result.shape != (values.shape[-1], len(TEMPORAL_FEATURE_NAMES)):
        raise RuntimeError(f"Unexpected temporal feature shape: {result.shape}")
    return result


def extract_top_variance_subcarriers(
    x: np.ndarray,
    count: int = TOP_VARIANCE_FEATURES,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Return ``[T,count]`` individually standardized high-variance streams.

    Selection and ordering are deterministic: descending temporal variance,
    then ascending original flattened subcarrier index for ties.
    """
    values = np.asarray(x, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"Expected [C,K,T], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Top-variance selection requires finite input")
    signals = values.reshape(-1, values.shape[-1])
    if signals.shape[0] < count:
        raise ValueError(f"Need at least {count} subcarriers, got {signals.shape[0]}")
    variances = signals.var(axis=1, dtype=np.float64)
    indices = np.arange(signals.shape[0])
    selected = np.lexsort((indices, -variances))[:count]
    chosen = signals[selected].astype(np.float64)
    means = chosen.mean(axis=1, keepdims=True)
    standard_deviations = chosen.std(axis=1, keepdims=True)
    if np.any(standard_deviations < epsilon):
        raise ValueError("Selected subcarrier contains a near-constant time series")
    standardized = (chosen - means) / standard_deviations
    return np.ascontiguousarray(standardized.T, dtype=np.float32)


def extract_interpolated_subcarriers(
    x: np.ndarray,
    count: int = INTERPOLATED_FEATURES,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Preserve subcarrier order and interpolate the frequency axis to ``count``.

    Each original subcarrier is standardized over time before interpolation.
    The output layout is ``[T,count]`` and every output position therefore has
    a consistent normalized frequency coordinate across samples.
    """
    values = np.asarray(x, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"Expected [C,K,T], got {values.shape}")
    if values.shape[0] != 1:
        raise ValueError(f"Expected one CSI channel, got {values.shape[0]}")
    if not np.isfinite(values).all():
        raise ValueError("Interpolation requires finite input")
    signals = values[0].astype(np.float64)
    means = signals.mean(axis=1, keepdims=True)
    standard_deviations = signals.std(axis=1, keepdims=True)
    valid = standard_deviations[:, 0] >= epsilon
    if int(valid.sum()) < 2:
        raise ValueError("Need at least two non-constant subcarriers")
    signals = (signals[valid] - means[valid]) / standard_deviations[valid]

    source_count = signals.shape[0]
    positions = np.linspace(0.0, source_count - 1, count)
    left = np.floor(positions).astype(np.int64)
    right = np.minimum(left + 1, source_count - 1)
    weight = (positions - left)[:, None]
    interpolated = signals[left] * (1.0 - weight) + signals[right] * weight
    return np.ascontiguousarray(interpolated.T, dtype=np.float32)
