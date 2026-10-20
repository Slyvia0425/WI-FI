"""Step 3a of preprocessing: time-series statistics features, ten numbers per channel (232 x 10 = 2,320).

Run from the repository root with the module's virtual environment:
    backend/modules/motion_source_recognition/.venv/bin/python -m backend.modules.motion_source_recognition.preprocessing.features [--limit N]

Per channel: mean, std, min, max, skewness, kurtosis, mean absolute first difference, dominant frequency (Hz), low-band energy (0.5-3 Hz),
high-band energy (3-20 Hz). The sampling rate is 100 Hz (500 steps = 5 s). Band energy is the sum of |FFT|^2 / N over the band, after the
channel mean is removed. No scaling is applied here: scalers are fitted on training samples only, later.

Writes outputs/prep/features_stats.npy (float32, rows in the order of manifest.csv), feature_ids.csv and features_stats_columns.json.
Only the CSI signal is used: user, activity, environment and device never enter the features.
"""
import argparse
import json
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

from .config import DATA, FEATURES, PREP_DIR, SAMPLE_SHAPE, data_root
from .windows import read_windows

SAMPLING_RATE_HZ = DATA["sampling_rate_hz"]
STATS = ["mean", "std", "min", "max", "skew", "kurtosis", "mean_abs_diff", "dominant_freq", "low_band_energy", "high_band_energy"]
LOW_BAND = tuple(FEATURES["low_band_hz"])
HIGH_BAND = tuple(FEATURES["high_band_hz"])
N_CHANNELS = SAMPLE_SHAPE[0]
CHUNK = 256


def statistics_features(x: np.ndarray) -> np.ndarray:
    """x: (samples, channels, time) -> (samples, len(STATS) * channels), statistic-major (all channels of 'mean', then 'std', ...)."""
    x = x.astype(np.float32, copy=False)
    centred = x - x.mean(axis=2, keepdims=True)
    spectrum = np.abs(np.fft.rfft(centred, axis=2)) ** 2 / x.shape[2]
    freqs = np.fft.rfftfreq(x.shape[2], d=1.0 / SAMPLING_RATE_HZ)
    low = (freqs >= LOW_BAND[0]) & (freqs < LOW_BAND[1])
    high = (freqs >= HIGH_BAND[0]) & (freqs < HIGH_BAND[1])
    with np.errstate(all="ignore"):  # a flat channel has no skewness or kurtosis; those become 0
        columns = [
            x.mean(axis=2),
            x.std(axis=2),
            x.min(axis=2),
            x.max(axis=2),
            stats.skew(x, axis=2),
            stats.kurtosis(x, axis=2),
            np.abs(np.diff(x, axis=2)).mean(axis=2),
            freqs[1:][spectrum[:, :, 1:].argmax(axis=2)],  # skip the 0 Hz bin
            spectrum[:, :, low].sum(axis=2),
            spectrum[:, :, high].sum(axis=2),
        ]
    out = np.stack(columns, axis=1)  # (samples, stats, channels)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).reshape(len(x), -1).astype(np.float32)


def column_names() -> list[str]:
    return [f"{s}_ch{c:03d}" for s in STATS for c in range(N_CHANNELS)]


def _work(paths: list[str]) -> np.ndarray:
    return statistics_features(read_windows(paths))


def run(limit: int | None = None, workers: int = 8) -> None:
    root = data_root()
    manifest = pd.read_csv(PREP_DIR / "manifest.csv")
    if limit:
        manifest = manifest.head(limit)
    paths = [str(root / p) for p in manifest["file_path"]]
    chunks = [paths[i:i + CHUNK] for i in range(0, len(paths), CHUNK)]

    target = PREP_DIR / ("features_stats_test.npy" if limit else "features_stats.npy")
    out = np.lib.format.open_memmap(target, mode="w+", dtype=np.float32, shape=(len(paths), len(STATS) * N_CHANNELS))
    start, row = time.time(), 0
    with Pool(workers) as pool:
        for part in pool.imap(_work, chunks):
            out[row:row + len(part)] = part
            row += len(part)
    out.flush()
    print(f"{row} samples x {out.shape[1]} features in {time.time() - start:.1f} s -> {target.name}")
    if not limit:
        manifest[["id"]].to_csv(PREP_DIR / "feature_ids.csv", index=False)
        (PREP_DIR / "features_stats_columns.json").write_text(json.dumps(column_names()))
        print("NaN or inf values:", int((~np.isfinite(out)).sum()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="process only the first N samples into a test file")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run(args.limit, args.workers)
