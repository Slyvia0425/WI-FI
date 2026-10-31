"""Confusion-focused error analysis for class imbalance and transitions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from .metrics import confusion_matrix


def top_confusions(
    y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[str], *, limit: int = 5
) -> list[dict]:
    matrix = confusion_matrix(y_true, y_pred, len(labels))
    rows: list[dict] = []
    for truth, row in enumerate(matrix):
        for predicted, count in enumerate(row):
            if truth != predicted and count:
                rows.append({"true": labels[truth], "predicted": labels[predicted], "count": count})
    rows.sort(key=lambda item: item["count"], reverse=True)
    return rows[:limit]


def class_support(y_true: Sequence[int], labels: Sequence[str]) -> dict[str, int]:
    counts = Counter(int(value) for value in y_true)
    return {label: counts.get(index, 0) for index, label in enumerate(labels)}
