"""Application service boundary for Module B."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from backend.shared.export import read_replay

from .config import AppConfig, load_config
from .domain import Prediction, prediction_from_scores
from .errors import ConfigurationError, InferenceError, ModelUnavailableError, ModelValidationError, ModuleBError
from .logging_utils import log_event, operation
from .preprocessing import FeatureStats, check_quality, orient_window, require_valid
from .resource import ResourceGuard, ResourceLimits
from .security import AccessPolicy
from .training import ModelSpec, build_model, count_parameters, predict_logits
from .transport import HttpTransport, LocalTransport, ResilientTransport, Transport


class HumanActivityService:
    def __init__(self, config: AppConfig | None = None, *, env: Mapping[str, str] | None = None) -> None:
        self.config = config or load_config(env=env)
        self.env = dict(os.environ if env is None else env)
        self.policy = build_policy(self.config)
        self.guard = ResourceGuard(
            ResourceLimits(
                max_batch_size=self.config.runtime.max_batch_size,
                max_concurrent_requests=self.config.runtime.max_concurrent_requests,
                max_input_megabytes=self.config.runtime.max_input_megabytes,
                requests_per_second=self.config.runtime.requests_per_second,
            )
        )
        self._model: torch.nn.Module | None = None
        self._stats: FeatureStats | None = None
        self._model_version: str | None = None
        self._transport: Transport | None = None

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "module": self.config.module_id,
            "mode": self.config.runtime.mode,
            "model_loaded": self._model is not None,
        }

    def replay(self) -> dict | None:
        return read_replay(self.config.module_id)

    def load_checkpoint(self, path: Path | None = None, *, actor: str = "system") -> None:
        checkpoint = path or self.default_checkpoint_path()
        resolved = self.policy.authorize_read(actor, checkpoint)
        if not resolved.exists():
            raise ModelUnavailableError("model checkpoint not found", {"path": str(resolved)})
        try:
            payload = torch.load(resolved, map_location="cpu", weights_only=True)
            metadata = payload["metadata"]
            model_meta = metadata["model"]
            spec = ModelSpec(
                name=model_meta["name"],
                feature_size=int(model_meta["feature_size"]),
                num_classes=int(model_meta["num_classes"]),
                hidden_dim=int(model_meta["hidden_dim"]),
                dropout=float(model_meta.get("dropout", self.config.model.dropout)),
                heads=int(model_meta.get("heads", self.config.model.transformer_heads)),
                layers=int(model_meta.get("layers", self.config.model.transformer_layers)),
            )
            model = build_model(spec)
            model.load_state_dict(payload["state_dict"])
            model.eval()
            stats_payload = metadata.get("feature_stats")
            if not stats_payload:
                raise ModelUnavailableError("checkpoint is missing feature statistics", {"path": str(resolved)})
            self._stats = FeatureStats.from_dict(stats_payload)
            if count_parameters(model) > self.config.model.max_parameters:
                raise ModelValidationError("checkpoint exceeds configured parameter budget")
            self._model = model
            self._model_version = str(metadata["model_version"])
        except ModuleBError:
            raise
        except Exception as exc:
            raise ModelUnavailableError(
                "checkpoint could not be loaded", {"path": str(resolved), "reason": type(exc).__name__}
            ) from exc

    def default_checkpoint_path(self) -> Path:
        return self.config.paths.output_root / "model.pt"

    def predict(self, window: np.ndarray, *, actor: str = "researcher") -> Prediction:
        array = np.asarray(window, dtype=np.float32)
        self.guard.validate_input_bytes(array.nbytes)
        self.policy.authorize_read(actor, self.config.paths.output_root)
        self._ensure_model()
        if self._model is None or self._stats is None or self._model_version is None:
            raise ModelUnavailableError("model artifacts are incomplete")
        report = check_quality(
            array,
            expected_feature_size=self.config.dataset.expected_feature_size,
            min_frames=self.config.dataset.min_frames,
            max_abs_amplitude=self.config.dataset.max_abs_amplitude,
        )
        require_valid(report, context="service.predict")
        oriented = orient_window(array, self.config.dataset.expected_feature_size)
        normalised = self._stats.transform(oriented)
        with self.guard.acquire(), operation("predict", quality=report.flag, actor=actor):
            logits = predict_logits(self._model, [normalised], batch_size=1, device="cpu")
        scores = _softmax(logits[0])
        probability_map = {label: float(scores[index]) for index, label in enumerate(self.config.model.class_ids)}
        return prediction_from_scores(
            probability_map,
            threshold=self.config.model.decision_threshold,
            stability=float(scores.max() - np.partition(scores, -2)[-2]),
            model_version=self._model_version,
            quality=report.flag,
        )

    def predict_batch(self, windows: Sequence[np.ndarray], *, actor: str = "researcher") -> list[Prediction]:
        self.guard.validate_batch(windows)
        return [self.predict(window, actor=actor) for window in windows]

    def handle(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        try:
            action = str(payload.get("action", "health"))
            if action == "health":
                return self.health()
            if action == "replay":
                replay = self.replay()
                return {"ok": replay is not None, "replay": replay}
            if action == "predict":
                window = np.asarray(payload["window"], dtype=np.float32)
                prediction = self.predict(window, actor=str(payload.get("actor", "system")))
                return {"ok": True, "prediction": prediction.to_frame(float(payload.get("t", 0.0)))}
            raise InferenceError("unknown service action", {"action": action})
        except ModuleBError as exc:
            log_event("service.error", code=exc.code, message=exc.message)
            return exc.to_payload()
        except Exception as exc:
            wrapped = InferenceError("unhandled service error", {"type": type(exc).__name__})
            log_event("service.error", code=wrapped.code, message=wrapped.message)
            return wrapped.to_payload()

    def transport(self) -> Transport:
        if self._transport is None:
            local = LocalTransport(self.handle)
            remote_endpoint = self.env.get(self.config.runtime.remote_endpoint_env)
            remote_token = self.env.get(self.config.runtime.remote_token_env)
            remote = HttpTransport(
                endpoint=remote_endpoint or "",
                token=remote_token,
                timeout_seconds=self.config.runtime.timeout_seconds,
                retries=self.config.runtime.retries,
                backoff_seconds=self.config.runtime.backoff_seconds,
            )
            if self.config.runtime.mode == "local":
                self._transport = local
            elif self.config.runtime.mode == "remote":
                if not remote_endpoint:
                    raise ConfigurationError(
                        "remote endpoint is not configured", {"env_name": self.config.runtime.remote_endpoint_env}
                    )
                self._transport = remote
            else:
                self._transport = ResilientTransport(
                    remote, local, allow_fallback=self.config.runtime.allow_local_fallback
                )
        return self._transport

    def _ensure_model(self) -> None:
        if self._model is None:
            self.load_checkpoint()


def build_policy(config: AppConfig) -> AccessPolicy:
    from backend.shared.paths import module_dir

    readable = tuple(path.resolve() for path in config.paths.data_roots if path.exists()) + (
        config.paths.output_root.resolve(),
    )
    writable = (config.paths.output_root.resolve(), module_dir(config.module_id).resolve())
    return AccessPolicy.build(
        roles=config.dataset.allowed_roles + ("system", "viewer"),
        readable_roots=readable or (config.paths.output_root.resolve(),),
        writable_roots=writable,
    )


def _softmax(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    shifted = values - values.max()
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()
