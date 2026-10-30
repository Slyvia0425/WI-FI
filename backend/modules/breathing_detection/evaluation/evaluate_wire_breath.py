"""Evaluate the frozen CSI-Bench LSTM on positive-only WiRe-Breath data."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from backend.modules.breathing_detection.preprocessing.wire_cache import (
    load_wire_breath_cache,
)
from backend.modules.breathing_detection.training.lstm import LightweightBreathingLSTM
from backend.modules.breathing_detection.training.train_lstm import choose_device


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def positive_only_metrics(
    probabilities: np.ndarray,
    assessable: np.ndarray,
    threshold: float,
) -> dict:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    assessable = np.asarray(assessable, dtype=bool)
    if probabilities.ndim != 1 or assessable.shape != probabilities.shape:
        raise ValueError("Probabilities and assessability must be matching vectors")
    selected = probabilities[assessable]
    detected = selected >= threshold
    return {
        "windows": int(len(probabilities)),
        "assessable": int(assessable.sum()),
        "not_assessable": int((~assessable).sum()),
        "assessable_rate": float(assessable.mean()) if len(assessable) else None,
        "breathing_detected": int(detected.sum()),
        "false_none": int((~detected).sum()),
        "breathing_recall": float(detected.mean()) if len(detected) else None,
        "probability_mean": float(selected.mean()) if len(selected) else None,
        "probability_p10": float(np.quantile(selected, 0.1)) if len(selected) else None,
        "probability_median": float(np.median(selected)) if len(selected) else None,
    }


def sliced_metrics(
    samples: list[dict],
    probabilities: np.ndarray,
    assessable: np.ndarray,
    threshold: float,
    field: str,
) -> dict[str, dict]:
    values = sorted({str(sample[field]) for sample in samples})
    result = {}
    for value in values:
        selected = np.asarray([str(sample[field]) == value for sample in samples])
        result[value] = positive_only_metrics(
            probabilities[selected], assessable[selected], threshold
        )
    return result


def false_none_analysis(
    samples: list[dict],
    probabilities: np.ndarray,
    assessable: np.ndarray,
    threshold: float,
) -> dict:
    indices = np.flatnonzero(assessable & (probabilities < threshold))
    recording_counts = Counter(samples[index]["recording_id"] for index in indices)
    cases = []
    for index in indices[np.argsort(probabilities[indices])]:
        sample = samples[int(index)]
        cases.append(
            {
                "index": int(index),
                "sample_id": sample["sample_id"],
                "recording_id": sample["recording_id"],
                "probability": float(probabilities[index]),
                "participant_id": sample["participant_id"],
                "room": sample["room"],
                "posture": sample["posture"],
                "environment": sample["environment"],
            }
        )
    return {
        "count": len(cases),
        "recordings_affected": len(recording_counts),
        "by_recording": dict(recording_counts.most_common()),
        "cases_lowest_probability_first": cases,
    }


@torch.no_grad()
def predict(model, sequences: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(sequences), batch_size):
        batch = np.asarray(sequences[start : start + batch_size], dtype=np.float32)
        logits = model(torch.from_numpy(batch).to(device))
        outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1401)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    cache = load_wire_breath_cache(args.cache_dir, args.count)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if checkpoint.get("representation") != "interpolated":
        raise ValueError("Frozen checkpoint does not use ordered interpolation")
    model = LightweightBreathingLSTM(
        input_size=int(checkpoint["input_size"]),
        hidden_size=int(checkpoint["hidden_size"]),
        projection_size=int(checkpoint["projection_size"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    probabilities = predict(model, cache.sequences, device, args.batch_size)
    assessable = np.asarray(cache.weights, dtype=bool)
    samples = cache.manifest["samples"]
    threshold = float(checkpoint["threshold"])
    report = {
        "evaluation": "frozen_positive_only_domain_evaluation",
        "dataset": "WiRe-Breath",
        "interpretation": (
            "All windows contain breathing; recall is measurable, specificity and "
            "macro-F1 are not. Quality-flagged windows are excluded from recall."
        ),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "threshold": threshold,
        "device": str(device),
        "overall": positive_only_metrics(probabilities, assessable, threshold),
        "flagged_windows_diagnostic": positive_only_metrics(
            probabilities, ~assessable, threshold
        ),
        "by_environment": sliced_metrics(
            samples, probabilities, assessable, threshold, "environment"
        ),
        "by_participant": sliced_metrics(
            samples, probabilities, assessable, threshold, "participant_id"
        ),
        "by_room": sliced_metrics(samples, probabilities, assessable, threshold, "room"),
        "by_posture": sliced_metrics(
            samples, probabilities, assessable, threshold, "posture"
        ),
        "false_none_analysis": false_none_analysis(
            samples, probabilities, assessable, threshold
        ),
        "quality_flags": dict(
            Counter(flag for sample in samples for flag in sample["quality_flags"])
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
