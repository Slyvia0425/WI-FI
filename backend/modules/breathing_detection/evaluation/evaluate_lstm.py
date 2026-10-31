"""One-way evaluation of a frozen breathing LSTM checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from backend.modules.breathing_detection.preprocessing.cache import (
    build_temporal_cache,
    load_temporal_cache,
)
from backend.modules.breathing_detection.preprocessing.data import BreathingDataset
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
    parser.add_argument("--build-cache", action="store_true")
    return parser.parse_args()


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics_by_subcarrier_count(
    labels: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    subcarrier_counts: np.ndarray,
    threshold: float,
) -> dict[str, dict]:
    result = {}
    for count in sorted(set(subcarrier_counts.tolist())):
        selected = (subcarrier_counts == count) & weights
        group_labels = labels[selected]
        metrics = binary_metrics(group_labels, probabilities[selected], threshold)
        empty_count = int(np.sum(group_labels == 0))
        sleep_count = int(np.sum(group_labels == 1))
        metrics["class_counts"] = {"empty": empty_count, "sleep": sleep_count}
        if empty_count == 0:
            metrics["specificity"] = None
            metrics["empty_room_false_positive_rate"] = None
        if empty_count == 0 or sleep_count == 0:
            metrics["macro_f1"] = None
            metrics["single_class_group"] = True
        else:
            metrics["single_class_group"] = False
        result[str(int(count))] = metrics
    return result


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    probabilities, labels, weights = [], [], []
    for sequence, batch_labels, batch_weights in loader:
        probabilities.append(torch.sigmoid(model(sequence.to(device))).cpu().numpy())
        labels.append(batch_labels.numpy().astype(np.int64))
        weights.append(batch_weights.numpy().astype(bool))
    return tuple(map(np.concatenate, (probabilities, labels, weights)))


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    source = BreathingDataset(args.data_root, "test_id")
    cache = (
        build_temporal_cache(source, args.cache_dir)
        if args.build_cache
        else load_temporal_cache(args.cache_dir, source.split, len(source))
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if checkpoint.get("representation") != "interpolated":
        raise ValueError("Frozen checkpoint does not use the interpolated representation")
    model = LightweightBreathingLSTM(
        input_size=int(checkpoint["input_size"]),
        hidden_size=int(checkpoint["hidden_size"]),
        projection_size=int(checkpoint["projection_size"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    threshold = float(checkpoint["threshold"])
    loader = loader_for_dataset(CachedTemporalDataset(cache), args.batch_size, False)
    probabilities, labels, weights = predict_all(model, loader, device)
    subcarrier_counts = np.asarray(
        [int(row["num_sub"]) for row in source._rows], dtype=np.int64
    )
    report = {
        "evaluation": "frozen_one_way_test",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "threshold": threshold,
        "device": str(device),
        "test": binary_metrics(labels[weights], probabilities[weights], threshold),
        "by_subcarrier_count": metrics_by_subcarrier_count(
            labels, probabilities, weights, subcarrier_counts, threshold
        ),
        "skipped": int((~weights).sum()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
