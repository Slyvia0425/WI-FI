"""Replay payload construction and contract-safe export."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.shared.contract import validate_replay
from backend.shared.export import write_replay
from backend.shared.paths import outputs_dir

from ..contract import validate_frame_values
from ..domain import CLASS_ID_TO_STATE, Prediction
from ..errors import ExportValidationError


@dataclass(frozen=True)
class ReplayArtifacts:
    module: str
    version: str
    model_version: str
    frames: list[dict]
    metrics: dict
    preprocessing: dict | None = None
    training: dict | None = None
    experiments: dict | None = None
    mock: bool = False

    def payload(self) -> dict:
        payload: dict[str, Any] = {
            "module": self.module,
            "version": self.version,
            "mock": bool(self.mock),
            "frames": self.frames,
            "metrics": self.metrics,
        }
        for key in ("preprocessing", "training", "experiments"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        payload["model_version"] = self.model_version
        return payload


def frames_from_predictions(predictions: Iterable[Prediction], *, window_seconds: float) -> list[dict]:
    frames = []
    for index, prediction in enumerate(predictions):
        t = round(index * window_seconds, 6)
        frame = prediction.to_frame(t)
        frame["activity_id"] = _activity_id(frame["activity"])
        frames.append(frame)
    return frames


def _activity_id(activity: str) -> int | None:
    for class_id, state in CLASS_ID_TO_STATE.items():
        if state.value == activity:
            return class_id
    return None


def export_artifacts(artifacts: ReplayArtifacts, *, bundle: bool = True, output_root: Path | None = None) -> dict:
    payload = artifacts.payload()
    try:
        problems = validate_replay(artifacts.module, payload) + validate_frame_values(payload)
        if problems:
            raise ExportValidationError("replay does not follow the contract", {"problems": problems})
        if output_root is not None:
            output_root.mkdir(parents=True, exist_ok=True)
            (output_root / "replay.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
            (output_root / "metrics.json").write_text(
                json.dumps(artifacts.metrics, indent=2, sort_keys=True), encoding="utf-8"
            )
        else:
            write_replay(artifacts.module, payload, bundle=bundle)
            target = outputs_dir(artifacts.module)
            target.mkdir(parents=True, exist_ok=True)
            (target / "metrics.json").write_text(
                json.dumps(artifacts.metrics, indent=2, sort_keys=True), encoding="utf-8"
            )
    except ExportValidationError:
        raise
    except Exception as exc:
        raise ExportValidationError("replay export failed", {"reason": str(exc)}) from exc
    return payload


def make_metrics_block(metrics: dict, labels: Sequence[str], confusion: Sequence[Sequence[int]]) -> dict:
    return {
        "table": [
            {"metric": "accuracy", "value": metrics.get("accuracy")},
            {"metric": "macro_f1", "value": metrics.get("macro_f1")},
        ],
        "confusion": {"labels": list(labels), "matrix": [list(row) for row in confusion]},
    }
