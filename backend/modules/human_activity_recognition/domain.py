"""Stable activity identities and typed prediction results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite

from .errors import InputValidationError


class ActivityState(str, Enum):
    JUMPING = "jumping"
    RUNNING = "running"
    SEATED_BREATHING = "seated-breathing"
    WALKING = "walking"
    WAVING_HAND = "wavinghand"
    TRANSITION = "transition"
    UNKNOWN = "unknown"


STABLE_STATES: tuple[ActivityState, ...] = (
    ActivityState.JUMPING,
    ActivityState.RUNNING,
    ActivityState.SEATED_BREATHING,
    ActivityState.WALKING,
    ActivityState.WAVING_HAND,
)
CLASS_ID_TO_STATE = {index: state for index, state in enumerate(STABLE_STATES)}
STATE_TO_CLASS_ID = {state: index for index, state in CLASS_ID_TO_STATE.items()}


@dataclass(frozen=True)
class Prediction:
    activity: ActivityState
    confidence: float
    stable: bool
    probabilities: Mapping[str, float] = field(default_factory=dict)
    stability: float = 0.0
    quality: str = "good"
    model_version: str = "unknown"

    def to_frame(self, t: float) -> dict:
        frame = {
            "t": float(t),
            "activity": self.activity.value,
            "confidence": round(float(self.confidence), 6),
            "stable": bool(self.stable),
            "probabilities": {key: round(float(value), 6) for key, value in self.probabilities.items()},
            "stability": round(float(self.stability), 6),
            "quality": self.quality,
            "model_version": self.model_version,
        }
        if self.activity is ActivityState.TRANSITION:
            frame["transition"] = True
        return frame


def validate_probabilities(values: Mapping[str, float], *, tolerance: float = 1e-3) -> dict[str, float]:
    if not values:
        raise InputValidationError("probabilities must not be empty")
    output: dict[str, float] = {}
    for key, value in values.items():
        number = float(value)
        if not isfinite(number) or number < 0.0 or number > 1.0:
            raise InputValidationError("probability must be finite and within [0, 1]", {"key": key, "value": value})
        output[str(key)] = number
    total = sum(output.values())
    if abs(total - 1.0) > tolerance:
        raise InputValidationError(
            "probabilities must sum to 1 within tolerance", {"sum": total, "tolerance": tolerance}
        )
    return output


def prediction_from_scores(
    scores: Mapping[str, float], *, threshold: float, stability: float, model_version: str, quality: str = "good"
) -> Prediction:
    probabilities = validate_probabilities(scores)
    ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    label, confidence = ordered[0]
    margin = confidence - (ordered[1][1] if len(ordered) > 1 else 0.0)
    stable = confidence >= threshold and margin >= threshold / 2 and quality == "good"
    try:
        activity = ActivityState(label)
    except ValueError:
        activity = ActivityState.UNKNOWN
    if not stable and activity in STABLE_STATES:
        activity = ActivityState.UNKNOWN
    return Prediction(
        activity=activity,
        confidence=confidence,
        stable=stable,
        probabilities=probabilities,
        stability=max(0.0, min(1.0, float(stability))),
        quality=quality,
        model_version=model_version,
    )
