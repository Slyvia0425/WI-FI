"""Serializable signal-only periodic-interference risk scorer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.modules.breathing_detection.preprocessing.features import FEATURE_NAMES


@dataclass(frozen=True)
class PeriodicInterferenceScorer:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    threshold: float

    @classmethod
    def from_dict(cls, payload: dict) -> "PeriodicInterferenceScorer":
        if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("Periodic risk feature schema does not match")
        scorer = cls(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            coefficients=np.asarray(payload["coefficients"], dtype=np.float64),
            intercept=float(payload["intercept"]),
            threshold=float(payload["threshold"]),
        )
        expected = (len(FEATURE_NAMES),)
        if any(array.shape != expected for array in (scorer.mean, scorer.scale, scorer.coefficients)):
            raise ValueError("Periodic risk parameter shape does not match feature schema")
        if np.any(scorer.scale <= 0) or not 0.0 <= scorer.threshold <= 1.0:
            raise ValueError("Invalid periodic risk scale or threshold")
        return scorer

    def score(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.shape[-1:] != (len(FEATURE_NAMES),):
            raise ValueError(f"Expected [...,{len(FEATURE_NAMES)}] features")
        logits = ((values - self.mean) / self.scale) @ self.coefficients + self.intercept
        logits = np.clip(logits, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def is_interference(self, features: np.ndarray) -> np.ndarray:
        return self.score(features) >= self.threshold
