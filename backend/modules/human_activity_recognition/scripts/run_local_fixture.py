#!/usr/bin/env python3
"""Train and export from the committed fixture without downloading csi-bench."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from backend.modules.human_activity_recognition.config import load_config
    from backend.modules.human_activity_recognition.experiments import load_dataset_split, run_single_model
    from backend.modules.human_activity_recognition.service import build_policy

    parser = argparse.ArgumentParser(description="Run Module B using the local synthetic csi-bench fixture")
    parser.add_argument("--model", default="cnn_1d")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    fixture = ROOT / "backend/modules/human_activity_recognition/fixtures/mini_csi_bench"
    output = ROOT / "backend/modules/human_activity_recognition/local_data/fixture_run"
    config = load_config(env={"HAR_DATA_ROOT": str(fixture), "HAR_OUTPUT_ROOT": str(output)})
    config = replace(
        config,
        model=replace(config.model, epochs=args.epochs, patience=1, hidden_dim=32, seeds=(args.seed,)),
    )
    bundle = load_dataset_split(config, build_policy(config))
    result = run_single_model(
        bundle,
        config,
        model_name=args.model,
        seed=args.seed,
        checkpoint_path=output / "model.pt",
        output_dir=output,
        mock=True,
    )
    print(json.dumps({"ok": True, "macro_f1": result["metrics"]["macro_f1"], "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
