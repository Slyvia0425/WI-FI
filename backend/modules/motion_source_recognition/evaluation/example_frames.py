"""Example output of the interface: real windows through the real pipeline (inference.Detector), taken from the VALIDATION rows of split 0 (no test row is used).

400 validation windows are drawn at random (seed 42). The examples are the first two windows of each true class in that draw, whatever came out, then the first two windows
whose answer was unknown, and two made-up broken windows (one with a dead channel, one with a missing value) to show the quality levels. Nothing is picked for looking good.
The true class is written next to each example only for the reader; it is not a field of the interface.

Run from the repository root (needs the dataset location, see the README):
    backend/modules/motion_source_recognition/.venv/bin/python -m backend.modules.motion_source_recognition.evaluation.example_frames
Writes outputs/evaluation/example_frames.json.
"""
import json

import numpy as np
import pandas as pd

from ..inference import Detector
from ..preprocessing.config import EVALUATION_DIR, PREP_DIR, data_root
from ..preprocessing.windows import read_windows


DRAW, PER_CLASS, SEED = 400, 2, 42


def run() -> dict:
    manifest, splits = pd.read_csv(PREP_DIR / "manifest.csv"), pd.read_csv(PREP_DIR / "splits.csv")
    rows = manifest[manifest["id"].isin(splits.loc[splits["split_0"] == "val", "id"])].sample(DRAW, random_state=SEED)
    windows = read_windows([str(data_root() / p) for p in rows["file_path"]])
    detector = Detector(split=0)
    frames = [detector(w) for w in windows]
    labels = rows["label"].tolist()
    picked, seen = [], {}
    for i, c in enumerate(labels):
        if seen.get(c, 0) < PER_CLASS:
            seen[c] = seen.get(c, 0) + 1
            picked.append(i)
    unknown_picked = [i for i, f in enumerate(frames) if f["label"] == "unknown" and i not in picked][:2]
    examples = [{"true_class": labels[i], "frame": frames[i]} for i in picked + unknown_picked]
    dead = windows[0].copy()
    dead[10] = 0.0
    broken = windows[1].copy()
    broken[5, 5] = np.nan
    examples += [{"true_class": "(made-up window with one dead channel)", "frame": detector(dead)}, {"true_class": "(made-up window with a missing value)", "frame": detector(broken)}]
    answers = pd.Series([f["label"] for f in frames]).value_counts().to_dict()
    return {"split": 0, "temperature": detector.temperature, "threshold": detector.threshold, "drawn_validation_windows": DRAW, "answers_over_the_draw": answers, "examples": examples}


if __name__ == "__main__":
    out = run()
    (EVALUATION_DIR / "example_frames.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "examples"}))
    for e in out["examples"]:
        print(e["true_class"], "->", json.dumps(e["frame"]))
