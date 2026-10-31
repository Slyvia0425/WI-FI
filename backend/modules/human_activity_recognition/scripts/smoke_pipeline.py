#!/usr/bin/env python3
"""Run the deterministic synthetic data-to-replay smoke path without a dataset download."""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from backend.modules.human_activity_recognition.config import load_config
    from backend.modules.human_activity_recognition.experiments import run_single_model, synthetic_split_bundle

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        config = load_config(env={"HAR_OUTPUT_ROOT": str(root), "HAR_DATA_ROOT": str(root / "data")})
        config = replace(config, model=replace(config.model, epochs=1, patience=1, hidden_dim=16, seeds=(7,)))
        bundle = synthetic_split_bundle(
            feature_size=config.dataset.expected_feature_size, frames=config.dataset.min_frames + 4
        )
        result = run_single_model(
            bundle,
            config,
            model_name="cnn_1d",
            seed=7,
            checkpoint_path=root / "model.pt",
            output_dir=root / "run",
            mock=True,
        )
    print(json.dumps({"ok": True, "macro_f1": result["metrics"]["macro_f1"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
