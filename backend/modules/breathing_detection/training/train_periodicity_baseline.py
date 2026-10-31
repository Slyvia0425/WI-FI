"""CLI for training the interpretable periodicity baseline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from backend.modules.breathing_detection.preprocessing.data import BreathingDataset
from backend.modules.breathing_detection.preprocessing.features import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    PeriodicityFeatureConfig,
    extract_periodicity_features,
)
from backend.modules.breathing_detection.training.periodicity_baseline import (
    LogisticBaseline,
    binary_metrics,
    select_macro_f1_threshold,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/BreathingDetection"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("backend/modules/breathing_detection/outputs/periodicity_baseline"),
    )
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-val", type=int)
    parser.add_argument("--top-k-subcarriers", type=int, default=64)
    parser.add_argument(
        "--no-cache", action="store_true", help="Recompute features even if a cache exists"
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate test_id after fitting and validation-only threshold selection",
    )
    return parser.parse_args()


def featurize(
    dataset: BreathingDataset,
    limit: int | None,
    config: PeriodicityFeatureConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], int]:
    count = min(len(dataset), limit) if limit is not None else len(dataset)
    x_rows, labels = [], []
    metadata = {key: [] for key in ("difficulty", "fan_on", "los", "participant_id", "device_id")}
    skipped = 0
    for index in range(count):
        sample = dataset[index]
        if not sample.assessable:
            skipped += 1
            continue
        x_rows.append(extract_periodicity_features(sample.x, config))
        labels.append(sample.label)
        for key in metadata:
            value = sample.metadata[key]
            metadata[key].append("unknown" if value is None else str(value))
    if not x_rows:
        raise RuntimeError("No assessable samples were available")
    return (
        np.stack(x_rows),
        np.asarray(labels, dtype=np.int64),
        {key: np.asarray(values) for key, values in metadata.items()},
        skipped,
    )


def cached_features(
    dataset: BreathingDataset,
    limit: int | None,
    config: PeriodicityFeatureConfig,
    cache_dir: Path,
    *,
    use_cache: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], int]:
    limit_name = "all" if limit is None else str(limit)
    cache_path = cache_dir / (
        f"features_v{FEATURE_VERSION}_{dataset.split}_n{limit_name}_k{config.top_k_subcarriers}.npz"
    )
    if use_cache and cache_path.is_file():
        payload = np.load(cache_path)
        metadata = {
            key.removeprefix("meta_"): payload[key]
            for key in payload.files
            if key.startswith("meta_")
        }
        return payload["x"], payload["y"], metadata, int(payload["skipped"])

    started = time.monotonic()
    x, y, metadata, skipped = featurize(dataset, limit, config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        x=x,
        y=y,
        skipped=np.asarray(skipped),
        **{f"meta_{key}": values for key, values in metadata.items()},
    )
    print(
        f"Cached {len(y)} {dataset.split} feature rows in "
        f"{time.monotonic() - started:.1f}s: {cache_path}"
    )
    return x, y, metadata, skipped


def metrics_by_slice(
    y: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    metadata: dict[str, np.ndarray],
) -> dict:
    result = {}
    for field in ("difficulty", "fan_on", "los"):
        field_result = {}
        for value in sorted(set(metadata[field])):
            if value == "unknown":
                continue
            mask = metadata[field] == value
            field_result[value] = binary_metrics(y[mask], probabilities[mask], threshold)
        result[field] = field_result
    return result


def main() -> None:
    args = parse_args()
    feature_config = PeriodicityFeatureConfig(
        top_k_subcarriers=args.top_k_subcarriers
    )
    train = BreathingDataset(args.data_root, "train_id")
    validation = BreathingDataset(args.data_root, "val_id")
    cache_dir = args.output_dir / "feature_cache"
    x_train, y_train, train_metadata, train_skipped = cached_features(
        train, args.max_train, feature_config, cache_dir, use_cache=not args.no_cache
    )
    x_val, y_val, val_metadata, val_skipped = cached_features(
        validation, args.max_val, feature_config, cache_dir, use_cache=not args.no_cache
    )

    model = LogisticBaseline().fit(x_train, y_train)
    validation_probabilities = model.predict_proba(x_val)
    threshold = select_macro_f1_threshold(y_val, validation_probabilities)
    metrics = {
        "train": binary_metrics(y_train, model.predict_proba(x_train), threshold),
        "validation": binary_metrics(y_val, validation_probabilities, threshold),
        "skipped": {"train": train_skipped, "validation": val_skipped},
        "limits": {"train": args.max_train, "validation": args.max_val},
        "class_counts": {
            "train": {"empty": int(np.sum(y_train == 0)), "sleep": int(np.sum(y_train == 1))},
            "validation": {"empty": int(np.sum(y_val == 0)), "sleep": int(np.sum(y_val == 1))},
        },
        "validation_by_slice": metrics_by_slice(
            y_val, validation_probabilities, threshold, val_metadata
        ),
    }
    if args.evaluate_test:
        test = BreathingDataset(args.data_root, "test_id")
        x_test, y_test, test_metadata, test_skipped = cached_features(
            test, None, feature_config, cache_dir, use_cache=not args.no_cache
        )
        test_probabilities = model.predict_proba(x_test)
        metrics["test"] = binary_metrics(y_test, test_probabilities, threshold)
        metrics["test_by_slice"] = metrics_by_slice(
            y_test, test_probabilities, threshold, test_metadata
        )
        metrics["skipped"]["test"] = test_skipped
        metrics["class_counts"]["test"] = {
            "empty": int(np.sum(y_test == 0)),
            "sleep": int(np.sum(y_test == 1)),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_payload = model.to_dict(FEATURE_NAMES, threshold)
    model_payload["feature_config"] = feature_config.to_dict()
    model_payload["feature_version"] = FEATURE_VERSION
    model_payload["coefficient_ranking"] = sorted(
        (
            {"feature": name, "coefficient": float(coefficient)}
            for name, coefficient in zip(FEATURE_NAMES, model.coef_)
        ),
        key=lambda item: abs(item["coefficient"]),
        reverse=True,
    )
    (args.output_dir / "model.json").write_text(
        json.dumps(model_payload, indent=2), encoding="utf-8"
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
