"""One real window per class, turned into features step by step, for the data stage of the website. Nothing is fitted here.

For each class the first window of a validation recording (the same recordings as the preview windows), channel 132 (the channel the first Random Forest tree asks about):
  picture    the whole window (every 2nd channel, every 5th time step, each channel scaled to its own mean and spread; one character per cell, see LEVEL)
  signal     the 500 amplitude values of the channel
  statistics the ten numbers of statistics_features() for this channel, and the power spectrum they are read from
  kernels    three MiniRocket-style kernels (length 9, six weights -1, three weights 2) at dilations 1, 4 and 16: the convolution output, the threshold, and the share of values above it
The thresholds are the median of the convolution over the four example windows. The real MiniRocket transformer picks about 10,000 kernel-dilation-threshold combinations from the training windows;
this is an illustration of one feature of it, not the transformer itself.

Run from the repository root:
    CSI_DATA_ROOT=... backend/modules/motion_source_recognition/.venv/bin/python -m backend.modules.motion_source_recognition.evaluation.feature_demo
Writes outputs/evaluation/feature_demo.json.
"""
import json

import numpy as np
import pandas as pd

from ..preprocessing.config import CLASSES, DATA, EVALUATION_DIR, FEATURES, PREP_DIR, SAMPLE_SHAPE, WEBSITE, data_root
from ..preprocessing.features import HIGH_BAND, LOW_BAND, SAMPLING_RATE_HZ, STATS, statistics_features
from ..preprocessing.windows import read_windows
from .export_replay import FRAMES_PER_CLASS, SEED


CHANNEL = WEBSITE["demo_channel"]
KERNELS = [{**k, "plus": tuple(k["plus"])} for k in WEBSITE["demo_kernels"]]
SHOW_CHANNELS, SHOW_STEPS = 2, 5
LEVEL = 6   # picture: each cell is one character, z-score in steps of 0.5 from -3 to 3, written as chr(48 + 6 + step)


def kernel_weights(plus) -> np.ndarray:
    w = -np.ones(9)
    w[list(plus)] = 2.0
    return w


def convolve(x: np.ndarray, w: np.ndarray, dilation: int) -> np.ndarray:
    """Only the positions where all 9 steps lie inside the window: len(x) - 8 * dilation values; value i belongs to the centre step i + 4 * dilation."""
    n = len(x) - 8 * dilation
    return sum(w[k] * x[k * dilation:k * dilation + n] for k in range(9))


def windows() -> dict:
    manifest, splits = pd.read_csv(PREP_DIR / "manifest.csv"), pd.read_csv(PREP_DIR / "splits.csv")
    val = manifest[manifest["id"].isin(splits.loc[splits["split_0"] == "val", "id"])]
    rng, out = np.random.default_rng(SEED), {}
    for c in CLASSES:
        group = rng.choice(val.loc[val["label"] == c, "group_id"].unique())
        row = val[val["group_id"] == group].sort_values("session").head(FRAMES_PER_CLASS).iloc[0]
        out[c] = read_windows([str(data_root() / row["file_path"])])[0]
    return out


def run() -> dict:
    wins = windows()
    signals = {c: w[CHANNEL].astype(float) for c, w in wins.items()}
    biases = {k["name"]: float(np.median(np.concatenate([convolve(s, kernel_weights(k["plus"]), k["dilation"]) for s in signals.values()]))) for k in KERNELS}
    freqs = np.fft.rfftfreq(500, d=1.0 / SAMPLING_RATE_HZ)
    keep = (freqs > 0) & (freqs <= 20)
    n_channels, n_steps = SAMPLE_SHAPE[:2]
    minirocket = np.load(PREP_DIR / "features_minirocket_split_0.npy", mmap_mode="r").shape[1]
    out = {"channel": CHANNEL, "sampling_rate_hz": SAMPLING_RATE_HZ, "steps": n_steps, "channels": n_channels, "antenna_links": DATA["antenna_links"],
           "statistics_per_channel": len(STATS), "statistics_features": len(STATS) * n_channels, "minirocket_features": int(minirocket), "minirocket_kernels": FEATURES["minirocket_kernels"], "bands": {"low": list(LOW_BAND), "high": list(HIGH_BAND)}, "frequencies": [round(float(f), 2) for f in freqs[keep]], "classes": []}
    for c, w in wins.items():
        s = signals[c]
        feats = statistics_features(w[None])[0].reshape(len(STATS), -1)[:, CHANNEL]
        power = np.abs(np.fft.rfft(s - s.mean())) ** 2 / len(s)
        sub = w[::SHOW_CHANNELS, ::SHOW_STEPS].astype(float)
        z = (sub - sub.mean(axis=1, keepdims=True)) / (sub.std(axis=1, keepdims=True) + 1e-9)
        kernels = []
        for k in KERNELS:
            conv = convolve(s, kernel_weights(k["plus"]), k["dilation"])
            kernels.append({"name": k["name"], "weights": kernel_weights(k["plus"]).astype(int).tolist(), "dilation": k["dilation"], "offset": 4 * k["dilation"], "output": [round(float(v), 2) for v in conv],
                            "threshold": round(biases[k["name"]], 2), "share_above": round(float((conv > biases[k["name"]]).mean()), 3)})
        out["classes"].append({"class": c, "picture": ["".join(chr(48 + LEVEL + int(v)) for v in row) for row in np.clip(np.round(z * 2), -LEVEL, LEVEL)], "signal": [round(float(v), 2) for v in s],
                               "statistics": {n: round(float(v), 4) for n, v in zip(STATS, feats)}, "power": [round(float(v), 3) for v in power[keep]], "kernels": kernels})
    return out


if __name__ == "__main__":
    (EVALUATION_DIR / "feature_demo.json").write_text(json.dumps(run()))
