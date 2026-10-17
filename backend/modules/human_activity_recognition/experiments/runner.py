"""Reproducible experiment runner for synthetic fixtures and real csi-bench data."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import AppConfig
from ..domain import STATE_TO_CLASS_ID, ActivityState, Prediction
from ..errors import DataValidationError
from ..evaluation import (
    ReplayArtifacts,
    classification_metrics,
    export_artifacts,
    frames_from_predictions,
    make_metrics_block,
)
from ..logging_utils import operation, write_audit_record
from ..preprocessing import (
    FeatureStats,
    SampleRecord,
    assert_no_split_leakage,
    check_quality,
    discover_records,
    orient_window,
    require_valid,
)
from ..preprocessing.hdf5_reader import read_csi_array
from ..security import AccessPolicy
from ..training import ModelSpec, TrainingSettings, predict_logits, train_model


@dataclass(frozen=True)
class WindowDataset:
    windows: list[np.ndarray]
    labels: list[int]
    records: list[SampleRecord]

    def __len__(self) -> int:
        return len(self.windows)


@dataclass(frozen=True)
class SplitBundle:
    train: WindowDataset
    validation: WindowDataset
    test: WindowDataset
    stats: FeatureStats


def _default_split(sample_id: str) -> str:
    bucket = int(sample_id[:8], 16) % 10
    return "test" if bucket == 9 else "val" if bucket == 8 else "train"


def load_dataset_split(config: AppConfig, policy: AccessPolicy, *, actor: str = "system") -> SplitBundle:
    records = discover_records(config, policy, actor=actor)
    normalised = [
        SampleRecord(
            sample_id=record.sample_id,
            path=record.path,
            session_id=record.session_id,
            user_id=record.user_id,
            environment_id=record.environment_id,
            device_id=record.device_id,
            activity=record.activity,
            split=record.split if record.split in {"train", "val", "test"} else _default_split(record.sample_id),
            metadata=record.metadata,
        )
        for record in records
    ]
    assert_no_split_leakage(normalised)
    buckets: dict[str, list[tuple[np.ndarray, int, SampleRecord]]] = {"train": [], "val": [], "test": []}
    for record in normalised:
        array = read_csi_array(record.path, config.dataset.hdf5_key_candidates)
        report = check_quality(
            array,
            expected_feature_size=config.dataset.expected_feature_size,
            min_frames=config.dataset.min_frames,
            max_abs_amplitude=config.dataset.max_abs_amplitude,
        )
        require_valid(report, context=record.sample_id)
        window = orient_window(array, config.dataset.expected_feature_size)
        state = ActivityState(record.activity)
        if state not in STATE_TO_CLASS_ID:
            raise DataValidationError(
                "dataset contains a decision state in supervised data",
                {"sample_id": record.sample_id, "activity": state.value},
            )
        buckets[record.split].append((window, STATE_TO_CLASS_ID[state], record))
    if not buckets["train"] or not buckets["val"] or not buckets["test"]:
        raise DataValidationError(
            "official splits must all be non-empty", {key: len(value) for key, value in buckets.items()}
        )
    stats = FeatureStats.fit(window for window, _, _ in buckets["train"])
    return SplitBundle(
        train=_dataset(buckets["train"]),
        validation=_dataset(buckets["val"]),
        test=_dataset(buckets["test"]),
        stats=stats,
    )


def _dataset(rows: Sequence[tuple[np.ndarray, int, SampleRecord]]) -> WindowDataset:
    return WindowDataset(
        windows=[row[0] for row in rows], labels=[row[1] for row in rows], records=[row[2] for row in rows]
    )


def synthetic_split_bundle(
    *, feature_size: int = 8, frames: int = 32, samples_per_class: int = 6, seed: int = 7
) -> SplitBundle:
    rng = np.random.default_rng(seed)
    windows: list[np.ndarray] = []
    labels: list[int] = []
    records: list[SampleRecord] = []
    for class_id in range(4):
        for index in range(samples_per_class):
            signal: np.ndarray = np.zeros((feature_size, frames), dtype=np.float32)
            signal[class_id % feature_size] = class_id + 1.0
            signal += rng.normal(0.0, 0.05, size=signal.shape).astype(np.float32)
            windows.append(signal)
            labels.append(class_id)
            records.append(
                SampleRecord(
                    sample_id=f"s{seed}-{class_id}-{index}",
                    path=Path(f"synthetic/{class_id}.h5"),
                    session_id=f"session-{index}",
                    user_id=f"user-{index}",
                    environment_id=f"env-{index % 2}",
                    device_id="synthetic",
                    activity=ActivityState(list(STATE_TO_CLASS_ID)[class_id]).value,
                    split="train" if index < 4 else "val" if index == 4 else "test",
                )
            )
    train = [i for i, record in enumerate(records) if record.split == "train"]
    val = [i for i, record in enumerate(records) if record.split == "val"]
    test = [i for i, record in enumerate(records) if record.split == "test"]
    stats = FeatureStats.fit(windows[i] for i in train)
    return SplitBundle(
        train=WindowDataset([windows[i] for i in train], [labels[i] for i in train], [records[i] for i in train]),
        validation=WindowDataset([windows[i] for i in val], [labels[i] for i in val], [records[i] for i in val]),
        test=WindowDataset([windows[i] for i in test], [labels[i] for i in test], [records[i] for i in test]),
        stats=stats,
    )


def model_spec_from_config(config: AppConfig, model_name: str, feature_size: int) -> ModelSpec:
    if model_name not in config.model.candidates:
        raise DataValidationError("model is not in the configured candidate list", {"model": model_name})
    return ModelSpec(
        name=model_name,
        feature_size=feature_size,
        num_classes=len(config.model.class_ids),
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
        heads=config.model.transformer_heads,
        layers=config.model.transformer_layers,
    )


def run_single_model(
    bundle: SplitBundle,
    config: AppConfig,
    *,
    model_name: str,
    seed: int,
    checkpoint_path: Path | None = None,
    output_dir: Path | None = None,
    mock: bool = False,
    version: str = "0.2.0",
) -> dict:
    feature_size = bundle.train.windows[0].shape[0]
    spec = model_spec_from_config(config, model_name, feature_size)
    settings = TrainingSettings(
        model=spec,
        batch_size=config.model.batch_size,
        epochs=config.model.epochs,
        learning_rate=config.model.learning_rate,
        weight_decay=config.model.weight_decay,
        patience=config.model.patience,
        max_parameters=config.model.max_parameters,
        seed=seed,
    )
    model_version = f"{model_name}-seed{seed}-{config.config_hash}"
    with operation("train_model", model=model_name, seed=seed, samples=len(bundle.train)):
        result = train_model(
            train_windows=bundle.train.windows,
            train_labels=bundle.train.labels,
            validation_windows=bundle.validation.windows,
            validation_labels=bundle.validation.labels,
            settings=settings,
            transform=bundle.stats,
            checkpoint_path=checkpoint_path,
            model_version=model_version,
            feature_stats=bundle.stats.to_dict(),
        )
    logits = predict_logits(result.model, bundle.test.windows, batch_size=config.runtime.max_batch_size)
    probabilities = _softmax(logits)
    predictions = [
        Prediction(
            activity=ActivityState(config.model.class_ids[int(index)])
            if config.model.class_ids[int(index)] in {state.value for state in ActivityState}
            else ActivityState.UNKNOWN,
            confidence=float(probabilities[row, int(index)]),
            stable=float(probabilities[row, int(index)]) >= 0.6,
            probabilities={label: float(probabilities[row, col]) for col, label in enumerate(config.model.class_ids)},
            stability=float(probabilities[row, int(index)] - np.partition(probabilities[row], -2)[-2]),
            quality="good",
            model_version=model_version,
        )
        for row, index in enumerate(probabilities.argmax(axis=1))
    ]
    metrics = classification_metrics(
        bundle.test.labels, probabilities.argmax(axis=1).tolist(), len(config.model.class_ids)
    )
    metrics_payload = metrics.to_dict()
    replay = ReplayArtifacts(
        module="human_activity_recognition",
        version=version,
        model_version=model_version,
        frames=frames_from_predictions(predictions, window_seconds=config.dataset.window_seconds),
        metrics=make_metrics_block(metrics_payload, config.model.class_ids, metrics.confusion),
        preprocessing={
            "tiles": [
                {"label": "train", "value": len(bundle.train.windows)},
                {"label": "validation", "value": len(bundle.validation.windows)},
                {"label": "test", "value": len(bundle.test.windows)},
            ]
        },
        training={
            "table": [
                {
                    "model": result.metadata["model"]["name"],
                    "parameters": result.parameter_count,
                    "best_epoch": result.best_epoch,
                }
            ]
        },
        experiments={
            "table": [
                {
                    "model": model_name,
                    "seed": seed,
                    "macro_f1": metrics_payload["macro_f1"],
                    "accuracy": metrics_payload["accuracy"],
                }
            ]
        },
        mock=mock,
    )
    payload = replay.payload()
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "evaluation.json").write_text(
            json.dumps({"metrics": metrics_payload, "training": result.metadata}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        export_artifacts(replay, bundle=False, output_root=output_dir)
        write_audit_record(
            output_dir / "audit.jsonl",
            "train_export",
            "system",
            model=model_name,
            seed=seed,
            model_version=model_version,
        )
    return {"metrics": metrics_payload, "replay": payload, "metadata": result.metadata}


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)
