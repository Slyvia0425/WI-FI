"""Stateful instability score for consecutive breathing windows."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TemporalInstabilityTracker:
    breathing_threshold: float
    window_size: int = 5
    min_history: int = 3
    _probabilities: deque[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.breathing_threshold < 1.0:
            raise ValueError("breathing_threshold must be in (0, 1)")
        if self.window_size < 2:
            raise ValueError("window_size must be at least 2")
        if not 2 <= self.min_history <= self.window_size:
            raise ValueError("min_history must be in [2, window_size]")
        self._probabilities = deque(maxlen=self.window_size)

    def reset(self) -> None:
        self._probabilities.clear()

    def update(self, probability: float) -> float | None:
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be finite and in [0, 1]")
        self._probabilities.append(float(probability))
        if len(self._probabilities) < self.min_history:
            return None

        values = np.asarray(self._probabilities)
        labels = values >= self.breathing_threshold
        flip_rate = float(np.mean(labels[1:] != labels[:-1]))
        signed_margin = np.where(
            labels,
            (values - self.breathing_threshold) / (1.0 - self.breathing_threshold),
            -(self.breathing_threshold - values) / self.breathing_threshold,
        )
        ambiguity = float(1.0 - np.mean(np.abs(signed_margin)))
        return float(np.clip(max(flip_rate, ambiguity), 0.0, 1.0))
