"""A.2 extension: spectral-shape features, and a reduced representation with one signal per antenna link instead of 232 channels.

Why: a fan turns at a steady speed, so its signal has a narrow, stable spectrum; a robot cleaner moves at a steady pace; people and pets move
irregularly, so their spectrum is broad. The statistics features only keep the strongest frequency and two coarse band energies, so they say
little about how peaked or how spread out the spectrum is. And the 4 antenna links are near copies of each other (58 subcarriers each), so a
single signal per link may carry the same information at a quarter of the cost (or less).

Per signal, ten numbers, all ratios (they do not depend on the signal size). The spectrum is |FFT|^2 of the mean-removed signal, restricted to
0.5-20 Hz (the 5 s window gives 0.2 Hz resolution):
  share of the power in 0.5-1, 1-2, 2-4, 4-8 and 8-20 Hz | spectral entropy (0 = one pure tone, 1 = flat) | spectral flatness |
  peak-to-mean ratio | spectral centroid (Hz) | spectral spread (Hz)

Writes to outputs/experiments/spectral/ (rows follow manifest.csv; features are statistic-major like features_stats.npy):
  spectral_channels.npy   (windows, 2320)  the ten numbers for each of the 232 channels
  link_stats.npy          (windows, 40)    the ten statistics of features.py for each link-mean signal
  link_spectral.npy       (windows, 40)    the ten spectral numbers for each link-mean signal

Run from the repository root (about 25 minutes for all windows, mostly reading files):
    nice -n 15 backend/modules/motion_source_recognition/.venv/bin/python -m backend.modules.motion_source_recognition.experiments.spectral_features [--limit N] [--workers 2]
"""
import argparse
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from ..preprocessing.config import DATA, EXPERIMENTS_DIR, PREP_DIR, data_root
from ..preprocessing.features import CHUNK, N_CHANNELS, statistics_features
from ..preprocessing.windows import read_windows
from ..preprocessing.physical import spectral_shape as spectral_features

OUT = EXPERIMENTS_DIR / "spectral"
LINKS = DATA["antenna_links"]


def link_means(x: np.ndarray) -> np.ndarray:
    """(samples, 232, time) -> (samples, 4, time): channels 58*i to 58*i+57 belong to link i."""
    return x.reshape(len(x), LINKS, N_CHANNELS // LINKS, x.shape[2]).mean(axis=2)


def _work(paths: list[str]) -> tuple:
    x = read_windows(paths)
    series = link_means(x).astype(np.float32)
    return spectral_features(x), statistics_features(series), spectral_features(series)


def run(limit: int | None, workers: int) -> None:
    manifest = pd.read_csv(PREP_DIR / "manifest.csv")
    if limit:
        manifest = manifest.head(limit)
    root = data_root()
    paths = [str(root / p) for p in manifest["file_path"]]
    chunks = [paths[i:i + CHUNK] for i in range(0, len(paths), CHUNK)]
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "_test" if limit else ""
    shapes = {"spectral_channels": (len(paths), 10 * N_CHANNELS), "link_stats": (len(paths), 10 * LINKS), "link_spectral": (len(paths), 10 * LINKS)}
    outs = {name: np.lib.format.open_memmap(OUT / f"{name}{suffix}.npy", mode="w+", dtype=np.float32, shape=shape) for name, shape in shapes.items()}
    start, row = time.time(), 0
    with Pool(workers) as pool:
        for parts in pool.imap(_work, chunks):
            for out, part in zip(outs.values(), parts):
                out[row:row + len(part)] = part
            row += len(parts[0])
            if (row // CHUNK) % 20 == 0:
                print(f"{row}/{len(paths)} windows, {time.time() - start:.0f} s", flush=True)
    for out in outs.values():
        out.flush()
    print(f"{row} windows in {time.time() - start:.0f} s; non-finite values:", {n: int((~np.isfinite(o)).sum()) for n, o in outs.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="process only the first N windows into *_test.npy files")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    run(args.limit, args.workers)
