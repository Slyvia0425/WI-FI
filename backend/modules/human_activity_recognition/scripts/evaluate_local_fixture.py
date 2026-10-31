#!/usr/bin/env python3
"""Compare configured model candidates on the committed local HDF5 fixture."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from backend.modules.human_activity_recognition.config import load_config
    from backend.modules.human_activity_recognition.experiments import load_dataset_split, run_single_model
    from backend.modules.human_activity_recognition.service import build_policy

    parser = argparse.ArgumentParser(description="Evaluate Module B candidates on the local synthetic fixture")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--models", default="cnn_1d,cnn_transformer")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    fixture = ROOT / "backend/modules/human_activity_recognition/fixtures/mini_csi_bench"
    output = ROOT / "backend/modules/human_activity_recognition/local_data/fixture_eval"
    config = load_config(env={"HAR_DATA_ROOT": str(fixture), "HAR_OUTPUT_ROOT": str(output)})
    config = replace(
        config,
        model=replace(config.model, epochs=args.epochs, patience=8, hidden_dim=16, seeds=tuple(seeds)),
    )
    bundle = load_dataset_split(config, build_policy(config))
    results: list[dict] = []
    for model_name in models:
        if model_name not in config.model.candidates:
            raise SystemExit(f"unsupported model: {model_name}")
        for seed in seeds:
            destination = output / model_name / f"seed-{seed}"
            result = run_single_model(
                bundle,
                config,
                model_name=model_name,
                seed=seed,
                checkpoint_path=destination / "model.pt",
                output_dir=destination,
                mock=True,
            )
            metrics = result["metrics"]
            results.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "confusion": metrics["confusion"],
                    "output_dir": str(destination),
                }
            )
    summary: dict[str, dict] = {}
    for model_name in models:
        rows = [row for row in results if row["model"] == model_name]
        summary[model_name] = {
            "accuracy_mean": float(np.mean([row["accuracy"] for row in rows])),
            "accuracy_std": float(np.std([row["accuracy"] for row in rows])),
            "macro_f1_mean": float(np.mean([row["macro_f1"] for row in rows])),
            "macro_f1_std": float(np.std([row["macro_f1"] for row in rows])),
            "seeds": seeds,
            "synthetic_fixture": True,
        }
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "warning": "Synthetic fixture results validate the pipeline; they are not csi-bench benchmark results.",
        "epochs": args.epochs,
        "runs": results,
        "summary": summary,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
