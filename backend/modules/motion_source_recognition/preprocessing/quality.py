"""Step 1b of preprocessing: open every h5 file once and record simple quality flags.

Run from the repository root (needs outputs/prep/manifest.csv from manifest.py):
    CSI_DATA_ROOT=/path/to/MotionSourceRecognition python3 -m backend.modules.motion_source_recognition.preprocessing.quality [--limit N]

Writes outputs/prep/quality.csv, one row per sample id, and outputs/prep/quality_report.json.
The flags are also what the module can pass to downstream modules as a data-quality marker.
Each row also has content_hash, a fingerprint of the whole window. Identical windows (614 pairs in this data, always inside one class)
share a hash; splits.py uses it so that no test window has an exact copy in training.
"""
import argparse
import hashlib
import json
import time
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd

from .config import H5_KEY, PREP_DIR, SAMPLE_SHAPE, data_root


def check_one(args: tuple) -> dict:
    sample_id, path = args
    row = {"id": sample_id, "read_ok": False}
    try:
        with h5py.File(path, "r") as f:
            x = f[H5_KEY][...]
    except Exception as exc:  # unreadable or missing key: record it, keep going
        row["error"] = f"{type(exc).__name__}: {exc}"[:120]
        return row
    row["content_hash"] = hashlib.sha1(x.tobytes()).hexdigest()
    finite = np.isfinite(x)
    channels = x.reshape(x.shape[0], -1)  # (232, 500)
    row.update(
        read_ok=True,
        shape_ok=x.shape == SAMPLE_SHAPE,
        n_nonfinite=int((~finite).sum()),
        n_zero_channels=int((np.abs(channels).max(axis=1) == 0).sum()),
        n_flat_channels=int((channels.std(axis=1) == 0).sum()),
        vmin=float(np.nanmin(x)),
        vmax=float(np.nanmax(x)),
        vmean=float(np.nanmean(x)),
    )
    return row


def run(limit: int | None = None, workers: int = 8) -> pd.DataFrame:
    root = data_root()
    manifest = pd.read_csv(PREP_DIR / "manifest.csv")
    if limit:
        manifest = manifest.sample(limit, random_state=0)
    jobs = [(i, str(root / p)) for i, p in zip(manifest["id"], manifest["file_path"])]

    start = time.time()
    with Pool(workers) as pool:
        rows = pool.map(check_one, jobs, chunksize=64)
    seconds = time.time() - start

    q = pd.DataFrame(rows)
    q["bad"] = ~q["read_ok"] | ~q.get("shape_ok", True).fillna(False) | (q.get("n_nonfinite", 0).fillna(0) > 0)
    report = {
        "samples_checked": len(q),
        "seconds": round(seconds, 1),
        "ms_per_sample_per_worker": round(seconds / len(q) * workers * 1000, 1),
        "unreadable": int((~q["read_ok"]).sum()),
        "wrong_shape": int((q["read_ok"] & ~q["shape_ok"].fillna(False)).sum()),
        "with_nonfinite_values": int((q["n_nonfinite"].fillna(0) > 0).sum()),
        "with_zero_channels": int((q["n_zero_channels"].fillna(0) > 0).sum()),
        "with_flat_channels": int((q["n_flat_channels"].fillna(0) > 0).sum()),
        "value_range": [float(q["vmin"].min()), float(q["vmax"].max())],
        "marked_bad": int(q["bad"].sum()),
        "windows_with_an_identical_twin": int(q["content_hash"].duplicated(keep=False).sum()),
    }
    PREP_DIR.mkdir(parents=True, exist_ok=True)
    if not limit:
        q.to_csv(PREP_DIR / "quality.csv", index=False)
        (PREP_DIR / "quality_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return q


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="check a random sample only (does not write files)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run(args.limit, args.workers)
