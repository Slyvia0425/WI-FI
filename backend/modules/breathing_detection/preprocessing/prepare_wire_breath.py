"""Prepare the ignored WiRe-Breath model-input cache and grouped folds."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .wire_breath import WireBreathDataset
from .wire_cache import build_wire_breath_cache, subject_holdout_folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/WiReBreath"))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("backend/modules/breathing_detection/outputs/wire_breath_cache"),
    )
    args = parser.parse_args()
    dataset = WireBreathDataset(args.data_root)
    cache = build_wire_breath_cache(dataset, args.cache_dir)
    folds = subject_holdout_folds(cache.manifest)
    fold_path = args.cache_dir / "subject_holdout_folds.json"
    fold_path.write_text(json.dumps(folds, indent=2), encoding="utf-8")
    samples = cache.manifest["samples"]
    print(
        json.dumps(
            {
                "windows": len(samples),
                "recordings": len({sample["recording_id"] for sample in samples}),
                "assessable": int(cache.weights.sum()),
                "environment": dict(Counter(x["environment"] for x in samples)),
                "subject": dict(Counter(x["participant_id"] for x in samples)),
                "folds": {name: {key: len(value) for key, value in fold.items()} for name, fold in folds.items()},
                "fold_manifest": str(fold_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
