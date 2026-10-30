"""Calibration of the model's probabilities: shared by the live module (inference.py), the experiments and the evaluation, so it lives here and not in one of their scripts.

to_logits / temper    divide the logits by one temperature (the class never changes), fitted on validation rows in evaluation/confidence_threshold.py
reliability           how far the confidence is from the real hit rate (calibration error), per confidence bin
"""
import numpy as np

from .preprocessing.config import EXPERIMENT_SETTINGS

EPS = 1e-30
EDGES, LABELS = EXPERIMENT_SETTINGS["calibration_edges"], EXPERIMENT_SETTINGS["calibration_labels"]


def to_logits(proba: np.ndarray) -> np.ndarray:
    """log-probabilities: for a softmax model they equal the logits up to one constant per window, which the softmax removes."""
    return np.log(np.clip(proba.astype(np.float64), EPS, 1.0))


def temper(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = logits / temperature
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def reliability(proba: np.ndarray, y: np.ndarray) -> dict:
    confidence, predicted = proba.max(axis=1), proba.argmax(axis=1)
    right = predicted == y
    bins = []
    for lo, hi, label in zip(EDGES[:-1], EDGES[1:], LABELS):
        mask = (confidence >= lo) & (confidence < hi)
        bins.append({"confidence": label, "share": round(float(mask.mean()), 3), "mean_confidence": round(float(confidence[mask].mean()), 3) if mask.any() else None,
                     "accuracy": round(float(right[mask].mean()), 3) if mask.any() else None})
    ece = sum(b["share"] * abs(b["accuracy"] - b["mean_confidence"]) for b in bins if b["share"])
    return {"ece": round(float(ece), 3), "mean_confidence": round(float(confidence.mean()), 3), "accuracy": round(float(right.mean()), 3), "bins": bins}
