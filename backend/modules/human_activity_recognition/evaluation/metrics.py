"""Dependency-light metrics used by tests and replay exports."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..errors import InputValidationError


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    precision: list[float]
    recall: list[float]
    f1: list[float]
    support: list[int]
    confusion: list[list[int]]

    def to_dict(self) -> dict:
        return {
            "accuracy": round(self.accuracy, 6),
            "macro_f1": round(self.macro_f1, 6),
            "precision": [round(v, 6) for v in self.precision],
            "recall": [round(v, 6) for v in self.recall],
            "f1": [round(v, 6) for v in self.f1],
            "support": self.support,
            "confusion": self.confusion,
        }


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> list[list[int]]:
    if len(y_true) != len(y_pred):
        raise InputValidationError("y_true and y_pred must have equal length")
    if not y_true:
        raise InputValidationError("metrics require at least one sample")
    matrix: np.ndarray = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, predicted in zip(y_true, y_pred, strict=False):
        if not 0 <= int(truth) < num_classes or not 0 <= int(predicted) < num_classes:
            raise InputValidationError(
                "class id out of range", {"truth": truth, "predicted": predicted, "num_classes": num_classes}
            )
        matrix[int(truth), int(predicted)] += 1
    return matrix.tolist()


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> ClassificationMetrics:
    matrix = np.asarray(confusion_matrix(y_true, y_pred, num_classes), dtype=np.float64)
    total = matrix.sum()
    precision: list[float] = []
    recall: list[float] = []
    f1: list[float] = []
    for index in range(num_classes):
        tp = matrix[index, index]
        fp = matrix[:, index].sum() - tp
        fn = matrix[index, :].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        precision.append(float(p))
        recall.append(float(r))
        f1.append(float(2 * p * r / (p + r)) if p + r else 0.0)
    return ClassificationMetrics(
        accuracy=float(np.trace(matrix) / total) if total else 0.0,
        macro_f1=float(np.mean(f1)),
        precision=precision,
        recall=recall,
        f1=f1,
        support=[int(row.sum()) for row in matrix],
        confusion=matrix.astype(int).tolist(),
    )


def expected_calibration_error(y_true: Sequence[int], probabilities: np.ndarray, *, bins: int = 10) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or len(y_true) != values.shape[0]:
        raise InputValidationError("probabilities must be a 2-D array with one row per label")
    if bins < 1:
        raise InputValidationError("bins must be positive")
    confidence = values.max(axis=1)
    prediction = values.argmax(axis=1)
    accuracy = prediction == np.asarray(y_true)
    error = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for low, high in zip(edges[:-1], edges[1:], strict=False):
        mask = (confidence > low) & (confidence <= high)
        if np.any(mask):
            error += float(mask.mean()) * abs(float(accuracy[mask].mean()) - float(confidence[mask].mean()))
    return error
