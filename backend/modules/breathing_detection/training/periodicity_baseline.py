"""Small NumPy logistic regression used by the interpretable baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.clip(np.asarray(logits, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-logits))


@dataclass
class LogisticBaseline:
    learning_rate: float = 0.05
    epochs: int = 800
    l2: float = 1e-3
    balanced: bool = True
    tolerance: float = 1e-8

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogisticBaseline":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim != 2 or y.shape != (x.shape[0],):
            raise ValueError("Expected X=[N,F] and y=[N]")
        if set(np.unique(y)) - {0.0, 1.0}:
            raise ValueError("LogisticBaseline supports binary labels 0/1")
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        z = (x - self.mean_) / self.scale_
        self.coef_ = np.zeros(z.shape[1], dtype=np.float64)
        self.intercept_ = 0.0

        if self.balanced:
            positives = max(float(y.sum()), 1.0)
            negatives = max(float((1.0 - y).sum()), 1.0)
            weights = np.where(y == 1.0, len(y) / (2 * positives), len(y) / (2 * negatives))
        else:
            weights = np.ones_like(y)

        previous_loss = np.inf
        for epoch in range(self.epochs):
            probabilities = sigmoid(z @ self.coef_ + self.intercept_)
            error = (probabilities - y) * weights
            grad_w = z.T @ error / len(y) + self.l2 * self.coef_
            grad_b = float(error.mean())
            rate = self.learning_rate / np.sqrt(1.0 + epoch / 100.0)
            self.coef_ -= rate * grad_w
            self.intercept_ -= rate * grad_b
            if epoch % 20 == 0 or epoch == self.epochs - 1:
                loss = float(
                    -np.mean(
                        weights
                        * (
                            y * np.log(np.maximum(probabilities, 1e-12))
                            + (1 - y) * np.log(np.maximum(1 - probabilities, 1e-12))
                        )
                    )
                    + 0.5 * self.l2 * np.dot(self.coef_, self.coef_)
                )
                if abs(previous_loss - loss) < self.tolerance:
                    break
                previous_loss = loss
        self.epochs_run_ = epoch + 1
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "coef_"):
            raise RuntimeError("Model has not been fitted")
        x = np.asarray(x, dtype=np.float64)
        z = (x - self.mean_) / self.scale_
        return sigmoid(z @ self.coef_ + self.intercept_)

    def to_dict(self, feature_names: tuple[str, ...], threshold: float) -> dict:
        return {
            "model": "periodicity_logistic_regression",
            "feature_names": list(feature_names),
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
            "coefficients": self.coef_.tolist(),
            "intercept": self.intercept_,
            "threshold": threshold,
            "training": {
                "learning_rate": self.learning_rate,
                "epochs_requested": self.epochs,
                "epochs_run": self.epochs_run_,
                "l2": self.l2,
                "balanced": self.balanced,
            },
        }


def binary_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    predicted = (np.asarray(probabilities) >= threshold).astype(np.int64)
    tp = int(np.sum((predicted == 1) & (y_true == 1)))
    tn = int(np.sum((predicted == 0) & (y_true == 0)))
    fp = int(np.sum((predicted == 1) & (y_true == 0)))
    fn = int(np.sum((predicted == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    negative_precision = tn / (tn + fn) if tn + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    negative_f1 = (
        2 * negative_precision * specificity / (negative_precision + specificity)
        if negative_precision + specificity
        else 0.0
    )
    return {
        "count": len(y_true),
        "threshold": float(threshold),
        "accuracy": (tp + tn) / len(y_true) if len(y_true) else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": (f1 + negative_f1) / 2,
        "specificity": specificity,
        "empty_room_false_positive_rate": 1.0 - specificity,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def select_macro_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Choose a validation-only threshold maximizing binary macro-F1."""
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    order = np.argsort(-probabilities)
    sorted_y = y_true[order]
    sorted_p = probabilities[order]
    tp = np.cumsum(sorted_y)
    fp = np.cumsum(1 - sorted_y)
    total_positive = int(sorted_y.sum())
    total_negative = len(sorted_y) - total_positive
    fn = total_positive - tp
    tn = total_negative - fp
    positive_denominator = 2 * tp + fp + fn
    negative_denominator = 2 * tn + fp + fn
    positive_f1 = np.divide(
        2 * tp,
        positive_denominator,
        out=np.zeros_like(tp, dtype=float),
        where=positive_denominator > 0,
    )
    negative_f1 = np.divide(
        2 * tn,
        negative_denominator,
        out=np.zeros_like(tn, dtype=float),
        where=negative_denominator > 0,
    )
    best = int(np.argmax((positive_f1 + negative_f1) / 2))
    return float(sorted_p[best])
