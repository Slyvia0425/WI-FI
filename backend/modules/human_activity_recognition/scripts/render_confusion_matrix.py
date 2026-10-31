#!/usr/bin/env python3
"""Render a saved Module B evaluation as an auditable confusion-matrix report."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LABELS = ["jumping", "running", "seated-breathing", "walking", "wavinghand"]
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write_matrix(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/predicted", *labels])
        for label, row in zip(labels, matrix.tolist(), strict=True):
            writer.writerow([label, *row])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--labels", nargs="+", default=LABELS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    parser.add_argument("--subset", choices=("all", "known_device"), default="all")
    args = parser.parse_args()

    payload = json.loads(args.evaluation.read_text(encoding="utf-8"))
    labels = list(args.labels)
    observations = payload.get("observations", {}).get(args.split, [])
    if observations:
        if args.subset == "known_device":
            observations = [item for item in observations if str(item.get("device", "")).upper() == "HP"]
        y_true = [int(item["true"]) for item in observations]
        y_pred = [int(item["predicted"]) for item in observations]
        from backend.modules.human_activity_recognition.evaluation.metrics import classification_metrics

        metrics = classification_metrics(y_true, y_pred, len(labels)).to_dict()
    else:
        metrics = (
            payload.get("subset_metrics", {}).get(f"{args.split}_{args.subset}")
            or payload.get("split_metrics", {}).get(args.split)
            or payload["metrics"]
        )
    matrix = np.asarray(metrics["confusion"], dtype=np.int64)
    if matrix.shape != (len(labels), len(labels)):
        raise ValueError(f"confusion matrix shape {matrix.shape} does not match {len(labels)} labels")

    output = args.output_dir or args.evaluation.parent / f"confusion_{args.split}_{args.subset}"
    output.mkdir(parents=True, exist_ok=True)
    _write_matrix(output / "confusion_matrix.csv", labels, matrix)

    with (output / "class_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        for index, label in enumerate(labels):
            writer.writerow(
                [
                    label,
                    metrics["precision"][index],
                    metrics["recall"][index],
                    metrics["f1"][index],
                    metrics["support"][index],
                ]
            )

    row_totals = matrix.sum(axis=1, keepdims=True)
    percentages = np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=float), where=row_totals != 0) * 100
    fig, ax = plt.subplots(figsize=(8.6, 7.2), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="True label",
        title=(
            f"Module B — {args.split}/{args.subset} — "
            f"Accuracy {metrics['accuracy']:.3f}, Macro-F1 {metrics['macro_f1']:.3f}"
        ),
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    threshold = matrix.max() / 2.0
    for row in range(len(labels)):
        for column in range(len(labels)):
            count = int(matrix[row, column])
            text = f"{count}\n{percentages[row, column]:.0f}%"
            color = "white" if count > threshold else "black"
            ax.text(column, row, text, ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(image, ax=ax, label="Samples")
    fig.savefig(output / "confusion_matrix.png", dpi=180)
    fig.savefig(output / "confusion_matrix.svg")
    plt.close(fig)

    summary = [
        f"# Module B — Real-data {args.split}/{args.subset} confusion matrix",
        "",
        f"- Accuracy: **{metrics['accuracy']:.4f}**",
        f"- Macro-F1: **{metrics['macro_f1']:.4f}**",
        f"- Samples: **{int(matrix.sum())}**",
        "",
        "## Confusion matrix",
        "",
        "| True \\ Predicted | " + " | ".join(labels) + " |",
        "|---|" + "---:|" * len(labels),
    ]
    for label, row in zip(labels, matrix.tolist(), strict=True):
        summary.append(f"| **{label}** | " + " | ".join(str(value) for value in row) + " |")
    summary += [
        "",
        "## Per-class metrics",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for index, label in enumerate(labels):
        summary.append(
            f"| {label} | {metrics['precision'][index]:.4f} | {metrics['recall'][index]:.4f} | "
            f"{metrics['f1'][index]:.4f} | {metrics['support'][index]} |"
        )
    (output / "evaluation_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"ok": True, "output_dir": str(output), "split": args.split, "subset": args.subset, **metrics},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
