"""Configuration loading and validation. No addresses, paths or credentials are hardcoded at call sites."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigurationError

MODULE_ID = "human_activity_recognition"
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


@dataclass(frozen=True)
class DatasetConfig:
    task_dir: str
    sample_root: str
    metadata_file: str
    label_mapping_file: str
    split_dir: str
    h5_glob: str
    hdf5_key_candidates: tuple[str, ...]
    expected_feature_size: int
    window_seconds: float
    min_frames: int
    split_group_fields: tuple[str, ...]
    max_abs_amplitude: float
    normalization: str
    allowed_roles: tuple[str, ...]


@dataclass(frozen=True)
class ModelConfig:
    default_name: str
    candidates: tuple[str, ...]
    class_ids: tuple[str, ...]
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    patience: int
    dropout: float
    decision_threshold: float
    stability_threshold: float
    hidden_dim: int
    transformer_heads: int
    transformer_layers: int
    max_parameters: int
    seeds: tuple[int, ...]
    augmentation_enabled: bool
    augmentation_max_shift: int
    augmentation_amplitude_scale: float
    augmentation_noise_std: float


@dataclass(frozen=True)
class SpectralConfig:
    top_features: int
    extra_models: int
    trees_per_model: int
    min_samples_leaf: int
    max_features: float
    histogram_weight: float
    histogram_iterations: int
    histogram_learning_rate: float
    histogram_leaf_nodes: int
    histogram_l2: float


@dataclass(frozen=True)
class RuntimeConfig:
    mode: str
    remote_endpoint_env: str
    remote_token_env: str
    timeout_seconds: float
    retries: int
    backoff_seconds: float
    requests_per_second: float
    max_batch_size: int
    max_concurrent_requests: int
    max_input_megabytes: int
    allow_local_fallback: bool


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    redact_keys: tuple[str, ...]


@dataclass(frozen=True)
class PathConfig:
    output_root: Path
    data_roots: tuple[Path, ...]


@dataclass(frozen=True)
class AppConfig:
    module_id: str
    dataset: DatasetConfig
    model: ModelConfig
    spectral: SpectralConfig
    runtime: RuntimeConfig
    logging: LoggingConfig
    paths: PathConfig
    source_path: Path
    secret_env_names: Mapping[str, str] = field(default_factory=dict)

    @property
    def data_root(self) -> Path | None:
        return self.paths.data_roots[0] if self.paths.data_roots else None

    @property
    def task_root(self) -> Path | None:
        return self.data_root / self.dataset.task_dir if self.data_root else None

    @property
    def config_hash(self) -> str:
        payload = asdict(self)
        payload.pop("secret_env_names", None)
        payload["paths"] = {
            "output_root": str(self.paths.output_root),
            "data_roots": [str(p) for p in self.paths.data_roots],
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]


def _tuple(value: Any, key: str) -> tuple:
    if not isinstance(value, list):
        raise ConfigurationError(f"{key} must be a list")
    return tuple(value)


def _positive(value: Any, key: str, *, allow_zero: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{key} must be numeric") from exc
    if number < 0 or (number == 0 and not allow_zero):
        raise ConfigurationError(f"{key} must be positive", {"value": value})
    return number


def _first_existing_path(names: tuple[str, ...], env: Mapping[str, str]) -> tuple[Path, ...]:
    roots: list[Path] = []
    for name in names:
        raw = env.get(name)
        if raw:
            roots.append(Path(raw).expanduser())
    return tuple(roots)


def load_config(path: str | Path | None = None, *, env: Mapping[str, str] | None = None) -> AppConfig:
    environment = dict(os.environ if env is None else env)
    source = Path(path or environment.get("HAR_CONFIG") or DEFAULT_CONFIG).expanduser()
    if not source.exists():
        raise ConfigurationError("configuration file not found", {"path": str(source)})
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError("configuration YAML is invalid", {"path": str(source), "reason": str(exc)}) from exc
    if not isinstance(raw, dict):
        raise ConfigurationError("configuration root must be a mapping", {"path": str(source)})
    if raw.get("module_id") != MODULE_ID:
        raise ConfigurationError("module_id does not match the module directory", {"module_id": raw.get("module_id")})

    ds = _require_mapping(raw.get("dataset"), "dataset")
    model = _require_mapping(raw.get("model"), "model")
    spectral = _require_mapping(raw.get("spectral"), "spectral")
    runtime = _require_mapping(raw.get("runtime"), "runtime")
    logging_cfg = _require_mapping(raw.get("logging"), "logging")
    paths = _require_mapping(raw.get("paths"), "paths")

    data_root_envs = tuple(str(v) for v in _tuple(paths.get("allowed_data_root_envs"), "paths.allowed_data_root_envs"))
    output_env = str(paths.get("output_root_env", "HAR_OUTPUT_ROOT"))
    output_root = (
        Path(environment[output_env]).expanduser().resolve()
        if environment.get(output_env)
        else (DEFAULT_CONFIG.parent / "outputs").resolve()
    )
    data_roots = _first_existing_path(data_root_envs, environment)
    mode = str(runtime.get("mode", "local")).lower()
    if mode not in {"local", "remote", "hybrid"}:
        raise ConfigurationError("runtime.mode must be local, remote or hybrid", {"mode": mode})

    dataset = DatasetConfig(
        task_dir=str(ds["task_dir"]),
        sample_root=str(ds.get("sample_root", ".")),
        metadata_file=str(ds["metadata_file"]),
        label_mapping_file=str(ds["label_mapping_file"]),
        split_dir=str(ds["split_dir"]),
        h5_glob=str(ds["h5_glob"]),
        hdf5_key_candidates=tuple(str(v) for v in _tuple(ds["hdf5_key_candidates"], "dataset.hdf5_key_candidates")),
        expected_feature_size=int(_positive(ds["expected_feature_size"], "dataset.expected_feature_size")),
        window_seconds=_positive(ds["window_seconds"], "dataset.window_seconds"),
        min_frames=int(_positive(ds["min_frames"], "dataset.min_frames")),
        split_group_fields=tuple(str(v) for v in _tuple(ds["split_group_fields"], "dataset.split_group_fields")),
        max_abs_amplitude=_positive(ds["max_abs_amplitude"], "dataset.max_abs_amplitude"),
        normalization=str(ds.get("normalization", "global_zscore")),
        allowed_roles=tuple(str(v) for v in _tuple(ds["allowed_roles"], "dataset.allowed_roles")),
    )
    if dataset.normalization not in {"global_zscore", "window_zscore"}:
        raise ConfigurationError(
            "dataset.normalization must be global_zscore or window_zscore",
            {"normalization": dataset.normalization},
        )
    augmentation = model.get("augmentation", {})
    if not isinstance(augmentation, Mapping):
        raise ConfigurationError("model.augmentation must be a mapping")
    model_cfg = ModelConfig(
        default_name=str(model["default_name"]),
        candidates=tuple(str(v) for v in _tuple(model["candidates"], "model.candidates")),
        class_ids=tuple(str(v) for v in _tuple(model["class_ids"], "model.class_ids")),
        batch_size=int(_positive(model["batch_size"], "model.batch_size")),
        epochs=int(_positive(model["epochs"], "model.epochs")),
        learning_rate=_positive(model["learning_rate"], "model.learning_rate"),
        weight_decay=_positive(model["weight_decay"], "model.weight_decay", allow_zero=True),
        patience=int(_positive(model["patience"], "model.patience")),
        dropout=_positive(model["dropout"], "model.dropout", allow_zero=True),
        decision_threshold=_positive(model["decision_threshold"], "model.decision_threshold"),
        stability_threshold=_positive(model["stability_threshold"], "model.stability_threshold", allow_zero=True),
        hidden_dim=int(_positive(model["hidden_dim"], "model.hidden_dim")),
        transformer_heads=int(_positive(model["transformer_heads"], "model.transformer_heads")),
        transformer_layers=int(_positive(model["transformer_layers"], "model.transformer_layers")),
        max_parameters=int(_positive(model["max_parameters"], "model.max_parameters")),
        seeds=tuple(int(v) for v in _tuple(model["seeds"], "model.seeds")),
        augmentation_enabled=bool(augmentation.get("enabled", False)),
        augmentation_max_shift=int(
            _positive(augmentation.get("max_shift", 8), "model.augmentation.max_shift", allow_zero=True)
        ),
        augmentation_amplitude_scale=_positive(
            augmentation.get("amplitude_scale", 0.05), "model.augmentation.amplitude_scale", allow_zero=True
        ),
        augmentation_noise_std=_positive(
            augmentation.get("noise_std", 0.005), "model.augmentation.noise_std", allow_zero=True
        ),
    )
    spectral_cfg = SpectralConfig(
        top_features=int(_positive(spectral["top_features"], "spectral.top_features")),
        extra_models=int(_positive(spectral["extra_models"], "spectral.extra_models")),
        trees_per_model=int(_positive(spectral["trees_per_model"], "spectral.trees_per_model")),
        min_samples_leaf=int(_positive(spectral["min_samples_leaf"], "spectral.min_samples_leaf")),
        max_features=_positive(spectral["max_features"], "spectral.max_features"),
        histogram_weight=_positive(spectral["histogram_weight"], "spectral.histogram_weight", allow_zero=True),
        histogram_iterations=int(_positive(spectral["histogram_iterations"], "spectral.histogram_iterations")),
        histogram_learning_rate=_positive(spectral["histogram_learning_rate"], "spectral.histogram_learning_rate"),
        histogram_leaf_nodes=int(_positive(spectral["histogram_leaf_nodes"], "spectral.histogram_leaf_nodes")),
        histogram_l2=_positive(spectral["histogram_l2"], "spectral.histogram_l2", allow_zero=True),
    )
    if spectral_cfg.histogram_weight > 1.0:
        raise ConfigurationError("spectral.histogram_weight must be between 0 and 1")
    runtime_cfg = RuntimeConfig(
        mode=mode,
        remote_endpoint_env=str(runtime["remote_endpoint_env"]),
        remote_token_env=str(runtime["remote_token_env"]),
        timeout_seconds=_positive(runtime["timeout_seconds"], "runtime.timeout_seconds"),
        retries=int(_positive(runtime["retries"], "runtime.retries", allow_zero=True)),
        backoff_seconds=_positive(runtime["backoff_seconds"], "runtime.backoff_seconds", allow_zero=True),
        requests_per_second=_positive(runtime["requests_per_second"], "runtime.requests_per_second"),
        max_batch_size=int(_positive(runtime["max_batch_size"], "runtime.max_batch_size")),
        max_concurrent_requests=int(_positive(runtime["max_concurrent_requests"], "runtime.max_concurrent_requests")),
        max_input_megabytes=int(_positive(runtime["max_input_megabytes"], "runtime.max_input_megabytes")),
        allow_local_fallback=bool(runtime["allow_local_fallback"]),
    )
    logging_config = LoggingConfig(
        level=str(logging_cfg.get("level", "INFO")),
        redact_keys=tuple(str(v) for v in _tuple(logging_cfg.get("redact_keys", []), "logging.redact_keys")),
    )
    if not data_roots:
        # A config without data roots is valid for contract-only and replay-only use.
        data_roots = ()
    secret_env_names = {"remote_token": runtime_cfg.remote_token_env}
    return AppConfig(
        module_id=MODULE_ID,
        dataset=dataset,
        model=model_cfg,
        spectral=spectral_cfg,
        runtime=runtime_cfg,
        logging=logging_config,
        paths=PathConfig(output_root=output_root, data_roots=data_roots),
        source_path=source,
        secret_env_names=secret_env_names,
    )


def _require_mapping(value: Any, key: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{key} must be a mapping")
    return value


def redacted_config(config: AppConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload.pop("secret_env_names", None)
    payload["source_path"] = str(config.source_path)
    payload["paths"] = {
        "output_root": str(config.paths.output_root),
        "data_roots": [str(path) for path in config.paths.data_roots],
    }
    return payload
