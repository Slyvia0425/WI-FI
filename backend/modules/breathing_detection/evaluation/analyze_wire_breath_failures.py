"""Signal-level analysis of WiRe-Breath false-none windows and gap policy."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

from backend.modules.breathing_detection.evaluation.evaluate_wire_breath import predict
from backend.modules.breathing_detection.preprocessing.wire_breath import WireBreathDataset
from backend.modules.breathing_detection.preprocessing.wire_cache import load_wire_breath_cache
from backend.modules.breathing_detection.training.lstm import LightweightBreathingLSTM
from backend.modules.breathing_detection.training.train_lstm import choose_device


TARGET_RECORDING = "Room2/Subject02/Sitting/Quiet/0816/Trial01"
GAP_THRESHOLDS_MS = (40, 50, 75, 100, 150, 250, 500, 1000)


def signal_features(sample, probability: float, threshold: float) -> dict:
    csi = np.asarray(sample.x[0], dtype=np.float64)
    reference = np.asarray(sample.respiratory_reference, dtype=np.float64)
    rate = float(sample.metadata["sampling_rate_hz"])
    window = np.hanning(csi.shape[1])
    frequencies = np.fft.rfftfreq(csi.shape[1], d=1.0 / rate)
    centered = csi - csi.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(centered * window, axis=1)) ** 2
    band = (frequencies >= 0.1) & (frequencies <= 0.7)
    comparison_band = (frequencies >= 0.1) & (frequencies <= 5.0)
    peaks = frequencies[band][np.argmax(power[:, band], axis=1)]

    reference_power = np.abs(
        np.fft.rfft((reference - reference.mean()) * window)
    ) ** 2
    reference_peak = float(frequencies[band][np.argmax(reference_power[band])])
    variances = np.var(csi, axis=1)
    means = np.mean(np.abs(csi), axis=1)
    relative_variability = np.sqrt(variances) / np.maximum(means, 1e-8)
    band_ratio = power[:, band].sum(axis=1) / np.maximum(
        power[:, comparison_band].sum(axis=1), 1e-12
    )
    values, counts = np.unique(peaks, return_counts=True)
    modal_peak = float(values[np.argmax(counts)])
    return {
        "probability": float(probability),
        "detected": bool(probability >= threshold),
        "reference_bpm": reference_peak * 60.0,
        "reference_std": float(reference.std()),
        "csi_variance_median": float(np.median(variances)),
        "csi_relative_variability_median": float(np.median(relative_variability)),
        "breathing_band_ratio_median": float(np.median(band_ratio)),
        "subcarrier_peak_mad_hz": float(np.median(np.abs(peaks - np.median(peaks)))),
        "subcarrier_modal_peak_hz": modal_peak,
        "subcarrier_modal_peak_fraction": float(np.max(counts) / len(peaks)),
        "subcarrier_gt_agreement_fraction": float(
            np.mean(np.abs(peaks - reference_peak) <= 0.1000001)
        ),
    }


def numeric_summary(rows: list[dict]) -> dict:
    result = {"count": len(rows)}
    if not rows:
        return result
    for key in rows[0]:
        if key in (
            "probability",
            "detected",
            "index",
            "recording_id",
            "participant_id",
            "posture",
            "environment",
        ):
            continue
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        result[key] = {
            "median": float(np.median(values)),
            "p10": float(np.quantile(values, 0.1)),
            "p90": float(np.quantile(values, 0.9)),
        }
    result["detection_rate"] = float(np.mean([row["detected"] for row in rows]))
    result["probability_median"] = float(np.median([row["probability"] for row in rows]))
    return result


def gap_scan(
    samples: list[dict], probabilities: np.ndarray, maximum_gaps: np.ndarray, threshold: float
) -> list[dict]:
    rows = []
    for gap_ms in GAP_THRESHOLDS_MS:
        eligible = maximum_gaps <= gap_ms / 1000.0
        detected = probabilities[eligible] >= threshold
        room1 = np.asarray([sample["room"] == "Room1" for sample in samples])
        room2 = ~room1
        rows.append(
            {
                "maximum_gap_ms": gap_ms,
                "assessable": int(eligible.sum()),
                "coverage": float(eligible.mean()),
                "breathing_recall": float(detected.mean()) if len(detected) else None,
                "breathing_output_rate_all_windows": float(
                    np.mean(eligible & (probabilities >= threshold))
                ),
                "room1_coverage": float(eligible[room1].mean()),
                "room2_coverage": float(eligible[room2].mean()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/WiReBreath"))
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    cache = load_wire_breath_cache(args.cache_dir, 1401)
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = LightweightBreathingLSTM(
        input_size=int(checkpoint["input_size"]),
        hidden_size=int(checkpoint["hidden_size"]),
        projection_size=int(checkpoint["projection_size"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    probabilities = predict(model, cache.sequences, device, args.batch_size)
    decision_threshold = float(checkpoint["threshold"])
    samples = cache.manifest["samples"]

    raw_dataset = WireBreathDataset(args.data_root, normalize=False)
    if len(raw_dataset) != len(samples):
        raise ValueError("Raw dataset and prepared cache are not aligned")
    time_cache: dict[Path, np.ndarray] = {}
    maximum_gaps = np.empty(len(samples), dtype=np.float64)
    feature_rows: list[dict] = []
    for index in range(len(raw_dataset)):
        raw_sample = raw_dataset[index]
        reference = raw_dataset._windows[index]
        if raw_sample.sample_id != samples[index]["sample_id"]:
            raise ValueError(f"Sample alignment mismatch at index {index}")
        if reference.path not in time_cache:
            payload = loadmat(reference.path, squeeze_me=True, variable_names=("time",))
            time_cache[reference.path] = np.asarray(payload["time"], dtype=np.float64).reshape(-1)
        time = time_cache[reference.path]
        target_end = reference.start_seconds + (
            raw_dataset.window_points - 1
        ) / raw_dataset.target_rate_hz
        overlap = (time[:-1] <= target_end) & (time[1:] >= reference.start_seconds)
        maximum_gaps[index] = float(np.max(np.diff(time)[overlap]))
        features = signal_features(raw_sample, probabilities[index], decision_threshold)
        features["index"] = index
        features["recording_id"] = raw_sample.recording_id
        features["participant_id"] = raw_sample.metadata["participant_id"]
        features["posture"] = raw_sample.metadata["posture"]
        features["environment"] = raw_sample.metadata["environment"]
        feature_rows.append(features)

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in feature_rows:
        is_target = row["recording_id"] == TARGET_RECORDING
        if is_target and not row["detected"]:
            groups["target_failed"].append(row)
        if is_target and row["detected"]:
            groups["target_successful"].append(row)
        if (
            row["participant_id"] == "02"
            and row["posture"] == "Sitting"
            and row["environment"] == "Interference"
        ):
            groups["same_subject_sitting_interference"].append(row)
        if row["posture"] == "Sitting" and not is_target and row["detected"]:
            groups["other_sitting_successful"].append(row)
        if row["detected"]:
            groups["all_successful"].append(row)

    room_gap_summary = {}
    for room in ("Room1", "Room2"):
        selected = np.asarray([sample["room"] == room for sample in samples])
        values = maximum_gaps[selected] * 1000.0
        room_gap_summary[room] = {
            "windows": int(selected.sum()),
            "median_ms": float(np.median(values)),
            "p90_ms": float(np.quantile(values, 0.9)),
            "p95_ms": float(np.quantile(values, 0.95)),
            "p99_ms": float(np.quantile(values, 0.99)),
            "maximum_ms": float(values.max()),
        }

    report = {
        "target_recording": TARGET_RECORDING,
        "decision_threshold": decision_threshold,
        "feature_definitions": {
            "breathing_band_hz": [0.1, 0.7],
            "spectral_comparison_band_hz": [0.1, 5.0],
            "subcarrier_gt_agreement_tolerance_hz": 0.1,
        },
        "signal_groups": {name: numeric_summary(rows) for name, rows in groups.items()},
        "room_timestamp_gap_distribution": room_gap_summary,
        "gap_threshold_scan": gap_scan(
            samples, probabilities, maximum_gaps, decision_threshold
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
