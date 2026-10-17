"""Command-line interface for Module B."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from .config import load_config, redacted_config
from .errors import ModuleBError
from .experiments import load_dataset_split, run_single_model, synthetic_split_bundle
from .logging_utils import operation
from .preprocessing import write_manifest
from .security import AccessPolicy
from .service import HumanActivityService, build_policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="human-activity-recognition", description="Module B development tasks")
    parser.add_argument(
        "--config", type=Path, default=None, help="path to config.yaml; defaults to HAR_CONFIG or module config"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("show-config", help="print non-secret configuration")

    manifest = sub.add_parser("manifest", help="discover csi-bench HDF5 records and write a stable manifest")
    manifest.add_argument("--output", type=Path, default=None)

    synthetic = sub.add_parser("train-synthetic", help="run a deterministic fixture through the full pipeline")
    synthetic.add_argument("--model", default=None)
    synthetic.add_argument("--seed", type=int, default=42)
    synthetic.add_argument("--epochs", type=int, default=2)
    synthetic.add_argument("--output-dir", type=Path, default=None)
    synthetic.add_argument("--checkpoint", type=Path, default=None)

    real = sub.add_parser("train", help="train on the configured csi-bench HumanActivityRecognition task")
    real.add_argument("--model", default=None)
    real.add_argument("--seed", type=int, default=42)
    real.add_argument("--output-dir", type=Path, default=None)
    real.add_argument("--checkpoint", type=Path, default=None)

    sub.add_parser("self-check", help="run import, contract and synthetic inference checks")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "show-config":
            print(json.dumps(redacted_config(config), indent=2, sort_keys=True))
            return 0
        if args.command == "manifest":
            policy = build_policy(config)
            records = load_records(config, policy)
            target = args.output or config.paths.output_root / "manifest.jsonl"
            with operation("manifest", output=str(target)):
                count = write_manifest(records, target)
            print(json.dumps({"ok": True, "records": count, "output": str(target)}, sort_keys=True))
            return 0
        if args.command == "train-synthetic":
            model = args.model or config.model.default_name
            model_config = replace(config.model, epochs=args.epochs)
            config = replace(config, model=model_config)
            bundle = synthetic_split_bundle(
                feature_size=config.dataset.expected_feature_size, frames=max(config.dataset.min_frames, 24)
            )
            destination = args.output_dir or config.paths.output_root / "synthetic"
            result = run_single_model(
                bundle,
                config,
                model_name=model,
                seed=args.seed,
                checkpoint_path=args.checkpoint or destination / "model.pt",
                output_dir=destination,
                mock=True,
            )
            print(
                json.dumps(
                    {"ok": True, "macro_f1": result["metrics"]["macro_f1"], "output_dir": str(destination)},
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "train":
            model = args.model or config.model.default_name
            destination = args.output_dir or config.paths.output_root / "run"
            policy = build_policy(config)
            bundle = load_dataset_split(config, policy)
            result = run_single_model(
                bundle,
                config,
                model_name=model,
                seed=args.seed,
                checkpoint_path=args.checkpoint or destination / "model.pt",
                output_dir=destination,
                mock=False,
            )
            print(
                json.dumps(
                    {"ok": True, "macro_f1": result["metrics"]["macro_f1"], "output_dir": str(destination)},
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "self-check":
            checks = self_check(config)
            print(json.dumps(checks, indent=2, sort_keys=True))
            return 0 if all(item["ok"] for item in checks.values()) else 1
        raise ModuleBError("INPUT_INVALID", "unknown command", {"command": args.command})
    except ModuleBError as exc:
        print(json.dumps(exc.to_payload(), indent=2, sort_keys=True))
        return 2


def load_records(config, policy: AccessPolicy):
    from .preprocessing import discover_records

    return discover_records(config, policy)


def self_check(config) -> dict:
    model = config.model.default_name
    tiny = replace(config, model=replace(config.model, epochs=1, patience=1))
    bundle = synthetic_split_bundle(feature_size=tiny.dataset.expected_feature_size, frames=tiny.dataset.min_frames + 4)
    result = run_single_model(bundle, tiny, model_name=model, seed=tiny.model.seeds[0], mock=True)
    service = HumanActivityService(tiny)
    return {
        "config": {"ok": True, "hash": tiny.config_hash, "source": str(tiny.source_path)},
        "synthetic_training": {"ok": result["metrics"]["macro_f1"] >= 0.0, "macro_f1": result["metrics"]["macro_f1"]},
        "model_contract": {"ok": "activity" in result["replay"]["frames"][0]},
        "service": {"ok": service.health()["ok"]},
    }


if __name__ == "__main__":
    sys.exit(main())
