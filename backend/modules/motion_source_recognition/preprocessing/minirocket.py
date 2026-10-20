"""Step 3b of preprocessing: MiniRocket features (9,996 per sample), fitted separately for each split.

Run from the repository root with the module's virtual environment (about 20 minutes per split):
    NUMBA_NUM_THREADS=6 backend/modules/motion_source_recognition/.venv/bin/python -m backend.modules.motion_source_recognition.preprocessing.minirocket [--split K]

For split K the transformer is fitted on a class-balanced random subset of that split's training samples only (MiniRocket uses the
signal, not the labels, to set its thresholds), then applied to every sample. The result for each split is
    outputs/prep/features_minirocket_split_K.npy   float32, rows in the order of manifest.csv
    outputs/prep/minirocket_split_K.pkl            the fitted transformer, needed to transform a new window later
A finished split leaves a .done marker and is skipped on the next run, so an interrupted run can simply be started again.
Only the CSI signal is used: user, activity, environment and device never enter the features.
"""
import argparse
import json
import pickle
import time

import numpy as np
import pandas as pd

from .config import FEATURES, PREP_DIR, data_root
from .splits import N_SPLITS
from .windows import read_windows

N_KERNELS = FEATURES["minirocket_kernels"]
CHUNK = 512


def run_split(k: int, threads: int, fit_per_class: int, limit: int | None = None) -> None:
    from aeon.transformations.collection.convolution_based import MiniRocket  # only needed here; installed in the module's venv

    suffix = "_test" if limit else ""
    target = PREP_DIR / f"features_minirocket_split_{k}{suffix}.npy"
    done = PREP_DIR / f"features_minirocket_split_{k}.done"
    if done.exists() and not limit:
        print(f"split_{k}: already done, skipped")
        return

    root = data_root()
    manifest = pd.read_csv(PREP_DIR / "manifest.csv")
    splits = pd.read_csv(PREP_DIR / "splits.csv")
    column = f"split_{k}"
    train = manifest[manifest["id"].isin(splits.loc[splits[column] == "train", "id"])]
    fit_rows = pd.concat([g.sample(min(fit_per_class, len(g)), random_state=k) for _, g in train.groupby("label")])

    start = time.time()
    fit_x = read_windows([str(root / p) for p in fit_rows["file_path"]])
    transformer = MiniRocket(n_kernels=N_KERNELS, random_state=k, n_jobs=threads)
    transformer.fit(fit_x)
    print(f"split_{k}: fitted on {len(fit_x)} training samples ({fit_rows['label'].value_counts().to_dict()}) in {time.time() - start:.0f} s")
    del fit_x
    (PREP_DIR / f"minirocket_split_{k}{suffix}.pkl").write_bytes(pickle.dumps(transformer))

    rows = manifest.head(limit) if limit else manifest
    paths = [str(root / p) for p in rows["file_path"]]
    first = transformer.transform(read_windows(paths[:2]))
    out = np.lib.format.open_memmap(target, mode="w+", dtype=np.float32, shape=(len(paths), first.shape[1]))
    start = time.time()
    for i in range(0, len(paths), CHUNK):
        out[i:i + CHUNK] = transformer.transform(read_windows(paths[i:i + CHUNK])).astype(np.float32)
        if (i // CHUNK) % 10 == 0:
            elapsed = time.time() - start
            print(f"split_{k}: {i + CHUNK:>6}/{len(paths)}  {elapsed / 60:.1f} min, about {elapsed / (i + CHUNK) * (len(paths) - i - CHUNK) / 60:.1f} min left", flush=True)
    out.flush()
    info = {"split": k, "features": int(out.shape[1]), "fit_samples_per_class": fit_rows["label"].value_counts().to_dict(),
            "seconds": round(time.time() - start), "nonfinite": int((~np.isfinite(out)).sum())}
    print(f"split_{k}: finished", info)
    if not limit:
        done.write_text(json.dumps(info))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=int, choices=range(N_SPLITS), help="one split only (default: all)")
    parser.add_argument("--threads", type=int, default=6, help="numba threads; keep a couple of cores free")
    parser.add_argument("--fit-per-class", type=int, default=FEATURES["minirocket_fit_per_class"])
    parser.add_argument("--limit", type=int, help="transform only the first N samples into a test file")
    args = parser.parse_args()
    for split in ([args.split] if args.split is not None else range(N_SPLITS)):
        run_split(split, args.threads, args.fit_per_class, args.limit)
