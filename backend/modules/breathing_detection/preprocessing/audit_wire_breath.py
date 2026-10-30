"""Audit the downloaded WiRe-Breath MATLAB release without modifying it."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.io import loadmat


REQUIRED_KEYS = {"csi_complex", "csi_abs", "time", "gt_signal", "gt_time", "metadata"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/WiReBreath"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def logical_parts(path: Path, root: Path) -> tuple[str, ...]:
    """Support normal directories and ZIPs extracted with literal backslashes."""
    relative = path.relative_to(root).as_posix().replace("\\", "/")
    return tuple(part for part in relative.split("/") if part)


def dominant_breathing_hz(signal: np.ndarray, time: np.ndarray) -> float | None:
    signal = np.asarray(signal, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    if len(signal) < 3 or len(time) != len(signal) or not np.all(np.diff(time) > 0):
        return None
    centered = signal - signal.mean()
    sample_rate = 1.0 / float(np.median(np.diff(time)))
    frequencies = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate)
    power = np.abs(np.fft.rfft(centered)) ** 2
    band = (frequencies >= 0.1) & (frequencies <= 0.7)
    if not band.any() or float(power[band].max()) <= 0:
        return None
    return float(frequencies[band][np.argmax(power[band])])


def audit(root: Path) -> dict:
    files = sorted(root.rglob("*.mat"), key=lambda path: logical_parts(path, root))
    if not files:
        raise FileNotFoundError(f"No MAT files found under {root}")

    counts = defaultdict(Counter)
    shape_counts = Counter()
    gt_length_counts = Counter()
    issues: list[dict] = []
    observed_rates, average_rates, durations, breathing_hz = [], [], [], []
    maximum_gaps_ms, large_gap_fractions = [], []
    amplitude_means, amplitude_stds = [], []
    recording_segments = Counter()
    breathing_by_environment = defaultdict(list)
    breathing_by_room = defaultdict(list)

    for path in files:
        try:
            payload = loadmat(path, squeeze_me=True, struct_as_record=False)
            missing = sorted(REQUIRED_KEYS - payload.keys())
            if missing:
                raise ValueError(f"missing fields: {missing}")
            csi_complex = np.asarray(payload["csi_complex"])
            csi_abs = np.asarray(payload["csi_abs"])
            time = np.asarray(payload["time"]).reshape(-1)
            gt_signal = np.asarray(payload["gt_signal"]).reshape(-1)
            gt_time = np.asarray(payload["gt_time"]).reshape(-1)
            metadata = payload["metadata"]

            if csi_complex.shape != csi_abs.shape:
                raise ValueError("complex/amplitude shape mismatch")
            if csi_abs.shape[0] != len(time):
                raise ValueError("CSI/time length mismatch")
            if len(gt_signal) != len(gt_time):
                raise ValueError("ground-truth/time length mismatch")
            arrays = (csi_complex.real, csi_complex.imag, csi_abs, time, gt_signal, gt_time)
            if not all(np.isfinite(array).all() for array in arrays):
                raise ValueError("non-finite values")
            if not np.all(np.diff(time) > 0) or not np.all(np.diff(gt_time) > 0):
                raise ValueError("timestamps are not strictly increasing")
            if not np.allclose(csi_abs, np.abs(csi_complex), rtol=1e-5, atol=1e-4):
                raise ValueError("csi_abs does not match abs(csi_complex)")

            room = str(metadata.room)
            subject = str(metadata.subject_id)
            posture = str(metadata.posture)
            environment = str(metadata.environment)
            counts["room"][room] += 1
            counts["subject"][subject] += 1
            counts["posture"][posture] += 1
            counts["environment"][environment] += 1
            counts["room_environment"][f"{room}/{environment}"] += 1
            counts["subject_environment"][f"Subject{subject}/{environment}"] += 1
            recording_id = "/".join(
                (
                    room,
                    f"Subject{subject}",
                    posture,
                    environment,
                    str(metadata.session_id),
                    f"Trial{int(metadata.trial_id):02d}",
                )
            )
            recording_segments[recording_id] += 1
            shape_counts[str(list(csi_abs.shape))] += 1
            gt_length_counts[str(len(gt_signal))] += 1

            time_differences = np.diff(time.astype(np.float64))
            median_interval = float(np.median(time_differences))
            observed_rates.append(1.0 / median_interval)
            average_rates.append((len(time) - 1) / float(time[-1] - time[0]))
            durations.append(float(time[-1] - time[0]))
            maximum_gaps_ms.append(float(time_differences.max() * 1000.0))
            large_gap_fractions.append(float(np.mean(time_differences > 2.0 * median_interval)))
            frequency = dominant_breathing_hz(gt_signal, gt_time)
            if frequency is not None:
                breathing_hz.append(frequency)
                breathing_by_environment[environment].append(frequency)
                breathing_by_room[room].append(frequency)
            amplitude_means.append(float(csi_abs.mean(dtype=np.float64)))
            amplitude_stds.append(float(csi_abs.std(dtype=np.float64)))

            logical = logical_parts(path, root)
            if room not in logical or str(metadata.file_name) != logical[-1]:
                raise ValueError("metadata path or file name mismatch")
        except Exception as error:
            issues.append({"file": str(path), "error": str(error)})

    def summary(values: list[float]) -> dict:
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": len(array),
            "min": float(array.min()) if len(array) else None,
            "median": float(np.median(array)) if len(array) else None,
            "max": float(array.max()) if len(array) else None,
        }

    return {
        "root": str(root),
        "mat_files": len(files),
        "literal_backslash_paths": sum("\\" in path.name for path in files),
        "counts": {name: dict(sorted(counter.items())) for name, counter in counts.items()},
        "csi_shape_counts": dict(shape_counts),
        "ground_truth_length_counts": dict(sorted(gt_length_counts.items(), key=lambda x: int(x[0]))),
        "recordings": {
            "unique": len(recording_segments),
            "segments_per_recording": dict(Counter(recording_segments.values())),
            "maximum_segments": max(recording_segments.values()),
        },
        "observed_csi_rate_hz": summary(observed_rates),
        "average_csi_rate_hz": summary(average_rates),
        "duration_seconds": summary(durations),
        "maximum_timestamp_gap_ms": summary(maximum_gaps_ms),
        "large_gap_fraction_by_file": summary(large_gap_fractions),
        "dominant_breathing_hz": summary(breathing_hz),
        "dominant_breathing_bpm": summary([value * 60.0 for value in breathing_hz]),
        "dominant_breathing_bpm_by_environment": {
            name: summary([value * 60.0 for value in values])
            for name, values in sorted(breathing_by_environment.items())
        },
        "dominant_breathing_bpm_by_room": {
            name: summary([value * 60.0 for value in values])
            for name, values in sorted(breathing_by_room.items())
        },
        "csi_amplitude_mean_by_file": summary(amplitude_means),
        "csi_amplitude_std_by_file": summary(amplitude_stds),
        "issues": issues,
    }


def main() -> None:
    args = parse_args()
    report = audit(args.data_root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
