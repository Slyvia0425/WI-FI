"""Validation-only error analysis for the frozen breathing LSTM."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from backend.modules.breathing_detection.evaluation.evaluate_lstm import predict_all
from backend.modules.breathing_detection.preprocessing.cache import load_temporal_cache
from backend.modules.breathing_detection.preprocessing.data import BreathingDataset
from backend.modules.breathing_detection.preprocessing.features import (
    FEATURE_NAMES,
    extract_periodicity_features,
)
from backend.modules.breathing_detection.training.lstm import LightweightBreathingLSTM
from backend.modules.breathing_detection.training.periodicity_baseline import binary_metrics
from backend.modules.breathing_detection.training.train_lstm import (
    CachedTemporalDataset,
    choose_device,
    loader_for_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/BreathingDetection"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--target-k", type=int, default=208)
    return parser.parse_args()


def empty_slice_summary(labels, probabilities, selected, threshold) -> dict:
    selected = np.asarray(selected, dtype=bool) & (labels == 0)
    values = probabilities[selected]
    false_positives = int(np.sum(values >= threshold))
    count = int(len(values))
    return {
        "empty_count": count,
        "false_positives": false_positives,
        "false_positive_rate": false_positives / count if count else None,
        "mean_probability": float(values.mean()) if count else None,
        "median_probability": float(np.median(values)) if count else None,
        "p90_probability": float(np.quantile(values, 0.9)) if count else None,
    }


def categorical_empty_slices(labels, probabilities, selected, values, threshold) -> dict:
    groups = defaultdict(list)
    for index, value in enumerate(values):
        if selected[index]:
            groups[str(value or "unknown")].append(index)
    result = {}
    for value, indices in groups.items():
        mask = np.zeros(len(labels), dtype=bool)
        mask[indices] = True
        summary = empty_slice_summary(labels, probabilities, mask, threshold)
        if summary["empty_count"]:
            result[value] = summary
    return dict(
        sorted(
            result.items(),
            key=lambda item: (
                -(item[1]["false_positive_rate"] or 0.0),
                -item[1]["empty_count"],
                item[0],
            ),
        )
    )


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    source = BreathingDataset(args.data_root, "val_id")
    cache = load_temporal_cache(args.cache_dir, source.split, len(source))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = LightweightBreathingLSTM(
        input_size=int(checkpoint["input_size"]),
        hidden_size=int(checkpoint["hidden_size"]),
        projection_size=int(checkpoint["projection_size"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    threshold = float(checkpoint["threshold"])
    loader = loader_for_dataset(CachedTemporalDataset(cache), args.batch_size, False)
    probabilities, labels, weights = predict_all(model, loader, device)
    rows = source._rows
    counts = np.asarray([int(row["num_sub"]) for row in rows])
    target = weights & (counts == args.target_k)

    by_k = {}
    for count in sorted(set(counts.tolist())):
        selected = weights & (counts == count)
        by_k[str(count)] = binary_metrics(
            labels[selected], probabilities[selected], threshold
        )
        by_k[str(count)]["class_counts"] = {
            "empty": int(np.sum(labels[selected] == 0)),
            "sleep": int(np.sum(labels[selected] == 1)),
        }

    fields = {
        "device": [row.get("device") for row in rows],
        "deployment": [row.get("deplyoment") or row.get("deployment") for row in rows],
        "difficulty": [row.get("difficulty") for row in rows],
    }
    false_positive = target & (labels == 0) & (probabilities >= threshold)
    target_empty_indices = np.flatnonzero(target & (labels == 0))
    periodicity = np.stack(
        [extract_periodicity_features(source[index].x) for index in target_empty_indices]
    )
    periodicity_is_fp = false_positive[target_empty_indices]
    feature_comparison = []
    for feature_index, name in enumerate(FEATURE_NAMES):
        fp_values = periodicity[periodicity_is_fp, feature_index]
        tn_values = periodicity[~periodicity_is_fp, feature_index]
        pooled_std = float(np.sqrt((fp_values.var() + tn_values.var()) / 2.0))
        feature_comparison.append(
            {
                "feature": name,
                "false_positive_median": float(np.median(fp_values)),
                "true_negative_median": float(np.median(tn_values)),
                "standardized_mean_difference": (
                    float((fp_values.mean() - tn_values.mean()) / pooled_std)
                    if pooled_std > 1e-12
                    else 0.0
                ),
            }
        )
    feature_comparison.sort(
        key=lambda row: -abs(row["standardized_mean_difference"])
    )
    top_indices = np.flatnonzero(false_positive)[
        np.argsort(-probabilities[false_positive])[:25]
    ]
    report = {
        "split": "val_id",
        "target_k": args.target_k,
        "threshold": threshold,
        "overall": binary_metrics(labels[weights], probabilities[weights], threshold),
        "by_k": by_k,
        "target_empty": empty_slice_summary(
            labels, probabilities, target, threshold
        ),
        "target_empty_by": {
            name: categorical_empty_slices(
                labels, probabilities, target, values, threshold
            )
            for name, values in fields.items()
        },
        "periodicity_feature_comparison": feature_comparison,
        "top_false_positives": [
            {
                "sample_id": rows[index]["id"],
                "probability": float(probabilities[index]),
                "device": rows[index].get("device"),
                "deployment": rows[index].get("deplyoment") or rows[index].get("deployment"),
                "difficulty": rows[index].get("difficulty"),
            }
            for index in top_indices
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
