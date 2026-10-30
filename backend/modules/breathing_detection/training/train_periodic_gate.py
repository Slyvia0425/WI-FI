"""Train a signal-only risk scorer for LSTM false-positive periodicity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from backend.modules.breathing_detection.evaluation.evaluate_lstm import predict_all
from backend.modules.breathing_detection.preprocessing.cache import load_temporal_cache
from backend.modules.breathing_detection.preprocessing.data import BreathingDataset
from backend.modules.breathing_detection.preprocessing.features import FEATURE_NAMES
from backend.modules.breathing_detection.training.lstm import LightweightBreathingLSTM
from backend.modules.breathing_detection.training.periodicity_baseline import (
    LogisticBaseline,
    binary_metrics,
)
from backend.modules.breathing_detection.training.train_lstm import (
    CachedTemporalDataset,
    choose_device,
    loader_for_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/BreathingDetection"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--temporal-cache-dir", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--max-correct-breathing-rejection", type=float, default=0.01)
    return parser.parse_args()


def load_feature_cache(cache_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    path = cache_dir / f"features_v1_{split}_nall_k64.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Periodicity feature cache not found: {path}")
    payload = np.load(path)
    return np.asarray(payload["x"], dtype=np.float64), np.asarray(payload["y"], dtype=np.int64)


def select_rejection_threshold(
    labels: np.ndarray,
    base_positive: np.ndarray,
    risks: np.ndarray,
    max_correct_rejection: float,
) -> tuple[float, dict]:
    if not 0.0 <= max_correct_rejection <= 1.0:
        raise ValueError("max_correct_rejection must be in [0, 1]")
    false_positive = base_positive & (labels == 0)
    correct_breathing = base_positive & (labels == 1)
    candidates = np.r_[np.nextafter(risks.max(), np.inf), np.unique(risks)]
    best = None
    for threshold in candidates:
        rejected = base_positive & (risks >= threshold)
        correct_rate = float(np.mean(rejected[correct_breathing])) if correct_breathing.any() else 0.0
        if correct_rate > max_correct_rejection + 1e-12:
            continue
        error_rate = float(np.mean(rejected[false_positive])) if false_positive.any() else 0.0
        candidate = (error_rate, -correct_rate, float(threshold), rejected)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    if best is None:
        raise RuntimeError("No feasible periodic risk threshold")
    error_rate, negative_correct_rate, threshold, rejected = best
    return threshold, {
        "false_positive_coverage": error_rate,
        "correct_breathing_rejection_rate": -negative_correct_rate,
        "rejected_count": int(rejected.sum()),
    }


def frozen_predictions(source, cache_dir, checkpoint, device, batch_size):
    cache = load_temporal_cache(cache_dir, source.split, len(source))
    loader = loader_for_dataset(CachedTemporalDataset(cache), batch_size, False)
    probabilities, labels, weights = predict_all(checkpoint, loader, device)
    return probabilities[weights], labels[weights]


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = LightweightBreathingLSTM(
        input_size=int(checkpoint["input_size"]),
        hidden_size=int(checkpoint["hidden_size"]),
        projection_size=int(checkpoint["projection_size"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    breathing_threshold = float(checkpoint["threshold"])

    sources = {
        split: BreathingDataset(args.data_root, split)
        for split in ("train_id", "val_id")
    }
    predictions = {
        split: frozen_predictions(
            source, args.temporal_cache_dir, model, device, args.batch_size
        )
        for split, source in sources.items()
    }
    features = {
        split: load_feature_cache(args.feature_cache_dir, split)
        for split in sources
    }
    for split in sources:
        probabilities, labels = predictions[split]
        x, feature_labels = features[split]
        if len(x) != len(probabilities) or not np.array_equal(feature_labels, labels):
            raise ValueError(f"Feature and temporal rows do not align for {split}")

    train_probability, train_labels = predictions["train_id"]
    x_train, _ = features["train_id"]
    train_positive = train_probability >= breathing_threshold
    risk_model = LogisticBaseline().fit(
        x_train[train_positive], (train_labels[train_positive] == 0).astype(np.int64)
    )

    val_probability, val_labels = predictions["val_id"]
    x_val, _ = features["val_id"]
    val_positive = val_probability >= breathing_threshold
    val_risk = risk_model.predict_proba(x_val)
    risk_threshold, selection = select_rejection_threshold(
        val_labels,
        val_positive,
        val_risk,
        args.max_correct_breathing_rejection,
    )
    rejected = val_positive & (val_risk >= risk_threshold)
    retained = ~rejected
    report = {
        "split": "val_id",
        "breathing_threshold": breathing_threshold,
        "periodic_interference_threshold": risk_threshold,
        "constraint": {
            "max_correct_breathing_rejection": args.max_correct_breathing_rejection
        },
        "selection": selection,
        "before_gate": binary_metrics(val_labels, val_probability, breathing_threshold),
        "after_gate_retained": binary_metrics(
            val_labels[retained], val_probability[retained], breathing_threshold
        ),
        "coverage": {
            "total": len(val_labels),
            "retained": int(retained.sum()),
            "rejected": int(rejected.sum()),
            "rejection_rate": float(rejected.mean()),
            "false_positives_rejected": int(np.sum(rejected & (val_labels == 0))),
            "correct_breathing_rejected": int(np.sum(rejected & (val_labels == 1))),
        },
        "training_base_positive": {
            "count": int(train_positive.sum()),
            "empty": int(np.sum(train_positive & (train_labels == 0))),
            "sleep": int(np.sum(train_positive & (train_labels == 1))),
        },
    }
    model_payload = risk_model.to_dict(FEATURE_NAMES, risk_threshold)
    model_payload.update(
        {
            "model": "periodic_interference_logistic_risk",
            "trained_on": "train_id frozen-LSTM-positive samples",
            "threshold_selected_on": "val_id",
            "breathing_threshold": breathing_threshold,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "model.json").write_text(
        json.dumps(model_payload, indent=2), encoding="utf-8"
    )
    (args.output_dir / "validation_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
