"""Reproducible spectral-feature ensemble for the five-class HAR task."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif

from ..config import AppConfig
from ..domain import ActivityState, Prediction
from ..errors import DataValidationError
from ..evaluation import (
    ReplayArtifacts,
    classification_metrics,
    export_artifacts,
    frames_from_predictions,
    make_metrics_block,
)
from ..logging_utils import operation, write_audit_record
from .runner import SplitBundle

BAND_EDGES = (1, 3, 6, 10, 15, 25, 40, 60, 90, 120, 160, 200, 250)
FEATURE_VERSION = "csi-band-power-v1"


def extract_spectral_features(windows: Sequence[np.ndarray]) -> np.ndarray:
    """Extract channel-wise band power and compact temporal statistics."""
    if not windows:
        raise DataValidationError("spectral feature extraction requires at least one window")
    arrays = [np.asarray(window, dtype=np.float32) for window in windows]
    feature_size, frames = arrays[0].shape
    if any(array.ndim != 2 or array.shape != (feature_size, frames) for array in arrays):
        raise DataValidationError("spectral feature extraction requires equal 2-D window shapes")
    common_size = frames // 2 + 1
    edges = tuple(edge for edge in BAND_EDGES if 1 <= edge < common_size) + (common_size,)
    band_count = len(edges) - 1
    output = np.empty((len(arrays), feature_size * (band_count + 4) + common_size), dtype=np.float32)
    for row, array in enumerate(arrays):
        centered = array - array.mean(axis=1, keepdims=True)
        power = np.abs(np.fft.rfft(centered, axis=1)) ** 2
        bands = np.stack(
            [power[:, low:high].mean(axis=1) for low, high in zip(edges[:-1], edges[1:], strict=True)],
            axis=1,
        )
        total = np.maximum(power.sum(axis=1, keepdims=True), 1e-9)
        proportions = power / total
        entropy = -(proportions * np.log(np.maximum(proportions, 1e-9))).sum(axis=1, keepdims=True) / np.log(
            power.shape[1]
        )
        channel_mean = centered.mean(axis=1, keepdims=True)
        channel_std = centered.std(axis=1, keepdims=True)
        zero_crossing = np.diff(np.signbit(centered), axis=1).mean(axis=1, keepdims=True)
        per_channel = np.concatenate(
            [np.log1p(bands), channel_mean, np.log1p(channel_std), zero_crossing, entropy], axis=1
        )
        common = np.log1p(power.mean(axis=0))
        output[row, : per_channel.size] = per_channel.ravel()
        output[row, per_channel.size :] = common
    if not np.isfinite(output).all():
        raise DataValidationError("spectral feature extraction produced non-finite values")
    return output


@dataclass
class SpectralEnsemble:
    selector: Any
    extra_models: tuple[Any, ...]
    histogram_model: Any | None
    histogram_weight: float
    labels: tuple[str, ...]
    feature_version: str = FEATURE_VERSION

    def _probabilities_from_features(self, features: np.ndarray) -> np.ndarray:
        selected = self.selector.transform(features)
        extra = np.mean([model.predict_proba(selected) for model in self.extra_models], axis=0)
        if self.histogram_model is None or self.histogram_weight == 0.0:
            return extra
        histogram = self.histogram_model.predict_proba(selected)
        return (1.0 - self.histogram_weight) * extra + self.histogram_weight * histogram

    def predict_proba(self, windows: Sequence[np.ndarray]) -> np.ndarray:
        return self._probabilities_from_features(extract_spectral_features(windows))

    def metadata(self) -> dict[str, Any]:
        return {
            "model_type": "spectral_ensemble",
            "feature_version": self.feature_version,
            "selected_features": int(self.selector.k),
            "extra_models": len(self.extra_models),
            "histogram_weight": self.histogram_weight,
            "labels": list(self.labels),
        }


def fit_spectral_ensemble(
    bundle: SplitBundle, config: AppConfig, *, seed: int
) -> tuple[SpectralEnsemble, dict[str, dict], dict[str, list[dict]]]:
    settings = config.spectral
    train_features = extract_spectral_features(bundle.train.windows)
    validation_features = extract_spectral_features(bundle.validation.windows)
    max_features = min(settings.top_features, train_features.shape[1])
    selector = SelectKBest(f_classif, k=max_features).fit(train_features, bundle.train.labels)
    selected_train = selector.transform(train_features)
    labels = config.model.class_ids
    extra_models = tuple(
        ExtraTreesClassifier(
            n_estimators=settings.trees_per_model,
            max_features=settings.max_features,
            min_samples_leaf=settings.min_samples_leaf,
            n_jobs=-1,
            random_state=seed + offset,
        ).fit(selected_train, bundle.train.labels)
        for offset in range(settings.extra_models)
    )
    histogram_model = None
    if settings.histogram_weight > 0.0:
        histogram_model = HistGradientBoostingClassifier(
            learning_rate=settings.histogram_learning_rate,
            max_iter=settings.histogram_iterations,
            max_leaf_nodes=settings.histogram_leaf_nodes,
            l2_regularization=settings.histogram_l2,
            random_state=seed,
        ).fit(selected_train, bundle.train.labels)
    model = SpectralEnsemble(
        selector=selector,
        extra_models=extra_models,
        histogram_model=histogram_model,
        histogram_weight=settings.histogram_weight,
        labels=labels,
    )
    split_metrics: dict[str, dict] = {}
    split_probabilities: dict[str, np.ndarray] = {}
    split_observations: dict[str, list[dict]] = {}
    for split_name, dataset, features in (
        ("validation", bundle.validation, validation_features),
        ("test", bundle.test, extract_spectral_features(bundle.test.windows)),
    ):
        probabilities = model._probabilities_from_features(features)
        split_probabilities[split_name] = probabilities
        predicted = probabilities.argmax(axis=1).tolist()
        split_metrics[split_name] = classification_metrics(dataset.labels, predicted, len(labels)).to_dict()
        split_observations[split_name] = [
            {
                "sample_id": record.sample_id,
                "true": int(truth),
                "predicted": int(prediction),
                "user": record.user_id,
                "environment": record.environment_id,
                "device": record.device_id,
                "session": record.session_id,
            }
            for record, truth, prediction in zip(dataset.records, dataset.labels, predicted, strict=True)
        ]
    return model, split_metrics, split_observations


def run_spectral_experiment(
    bundle: SplitBundle,
    config: AppConfig,
    *,
    seed: int,
    output_dir: Path,
    version: str = "0.3.0",
) -> dict[str, Any]:
    model_version = f"spectral-ensemble-{FEATURE_VERSION}-seed{seed}-{config.config_hash}"
    with operation("train_spectral_ensemble", seed=seed, samples=len(bundle.train)):
        model, split_metrics, split_observations = fit_spectral_ensemble(bundle, config, seed=seed)
    test_probabilities = model.predict_proba(bundle.test.windows)
    predictions = [
        Prediction(
            activity=ActivityState(model.labels[int(index)]),
            confidence=float(test_probabilities[row, int(index)]),
            stable=float(test_probabilities[row, int(index)]) >= config.model.decision_threshold,
            probabilities={label: float(test_probabilities[row, column]) for column, label in enumerate(model.labels)},
            stability=float(test_probabilities[row, int(index)] - np.partition(test_probabilities[row], -2)[-2]),
            quality="good",
            model_version=model_version,
        )
        for row, index in enumerate(test_probabilities.argmax(axis=1))
    ]
    metrics_payload = split_metrics["test"]
    replay = ReplayArtifacts(
        module="human_activity_recognition",
        version=version,
        model_version=model_version,
        frames=frames_from_predictions(predictions, window_seconds=config.dataset.window_seconds),
        metrics=make_metrics_block(metrics_payload, model.labels, metrics_payload["confusion"]),
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
                    "model": "spectral_ensemble",
                    "parameters": int(model.selector.k) * len(model.extra_models),
                    "best_epoch": 0,
                }
            ]
        },
        experiments={
            "table": [
                {
                    "model": "spectral_ensemble",
                    "seed": seed,
                    "macro_f1": metrics_payload["macro_f1"],
                    "accuracy": metrics_payload["accuracy"],
                }
            ]
        },
        mock=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")
    evaluation_payload = {
        "metrics": metrics_payload,
        "split_metrics": split_metrics,
        "observations": split_observations,
        "model": model.metadata(),
        "model_version": model_version,
        "training": {
            "seed": seed,
            "train_samples": len(bundle.train.windows),
            "validation_samples": len(bundle.validation.windows),
            "test_samples": len(bundle.test.windows),
        },
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(evaluation_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    export_artifacts(replay, bundle=False, output_root=output_dir)
    write_audit_record(
        output_dir / "audit.jsonl",
        "train_spectral_export",
        "system",
        seed=seed,
        model_version=model_version,
    )
    return {"metrics": metrics_payload, "model_version": model_version, "output_dir": str(output_dir)}


__all__ = [
    "BAND_EDGES",
    "FEATURE_VERSION",
    "SpectralEnsemble",
    "extract_spectral_features",
    "fit_spectral_ensemble",
    "run_spectral_experiment",
]
