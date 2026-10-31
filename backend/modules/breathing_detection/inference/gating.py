"""Transparent, ordered gates around the breathing model output."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite


class GateReason(StrEnum):
    LOW_QUALITY = "low_quality"
    MOTION = "motion"
    FAN = "fan"
    PERIODIC_INTERFERENCE = "periodic_interference"
    TEMPORAL_INSTABILITY = "temporal_instability"
    BREATHING_EVIDENCE = "breathing_evidence"
    NO_BREATHING_EVIDENCE = "no_breathing_evidence"


@dataclass(frozen=True)
class GateConfig:
    breathing_threshold: float = 0.054542455822229385
    motion_threshold: float = 0.8
    fan_threshold: float = 0.8
    periodic_interference_threshold: float = 0.8
    temporal_instability_threshold: float = 0.8

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            _validate_score(name, value)


@dataclass(frozen=True)
class GateInputs:
    breathing_probability: float
    quality_assessable: bool = True
    quality_flags: tuple[str, ...] = ()
    motion_probability: float | None = None
    fan_probability: float | None = None
    periodic_interference_risk: float | None = None
    temporal_instability: float | None = None

    def __post_init__(self) -> None:
        _validate_score("breathing_probability", self.breathing_probability)
        for name in (
            "motion_probability",
            "fan_probability",
            "periodic_interference_risk",
            "temporal_instability",
        ):
            value = getattr(self, name)
            if value is not None:
                _validate_score(name, value)
        if not isinstance(self.quality_assessable, bool):
            raise TypeError("quality_assessable must be bool")


@dataclass(frozen=True)
class GateDecision:
    status: str
    confidence: float
    reason: GateReason
    breathing_probability: float
    triggered_score: float | None = None
    quality_flags: tuple[str, ...] = ()

    @property
    def assessable(self) -> bool:
        return self.status != "not_assessable"

    def to_frame(self, t: float) -> dict:
        """Return the current public frame contract without exposing internals."""
        return {"t": float(t), "status": self.status, "confidence": self.confidence}


def _validate_score(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a numeric score")
    if not isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")


def _not_assessable(
    inputs: GateInputs,
    reason: GateReason,
    score: float,
) -> GateDecision:
    return GateDecision(
        status="not_assessable",
        confidence=float(score),
        reason=reason,
        breathing_probability=float(inputs.breathing_probability),
        triggered_score=float(score),
        quality_flags=tuple(inputs.quality_flags),
    )


def _decision_margin(probability: float, threshold: float) -> float:
    """Scale distance from the frozen threshold to [0,1] on either side."""
    if probability >= threshold:
        return (probability - threshold) / max(1.0 - threshold, 1e-12)
    return (threshold - probability) / max(threshold, 1e-12)


def decide_breathing_status(
    inputs: GateInputs,
    config: GateConfig | None = None,
) -> GateDecision:
    """Apply ordered gates, then classify with the frozen breathing threshold.

    A missing optional risk score means that evidence source is unavailable,
    not that its risk is zero. The periodic risk input is intentionally not
    computed here; it must be calibrated using train/validation data only.
    """
    config = config or GateConfig()
    if not inputs.quality_assessable or inputs.quality_flags:
        return _not_assessable(inputs, GateReason.LOW_QUALITY, 1.0)

    ordered_gates = (
        (inputs.motion_probability, config.motion_threshold, GateReason.MOTION),
        (inputs.fan_probability, config.fan_threshold, GateReason.FAN),
        (
            inputs.periodic_interference_risk,
            config.periodic_interference_threshold,
            GateReason.PERIODIC_INTERFERENCE,
        ),
        (
            inputs.temporal_instability,
            config.temporal_instability_threshold,
            GateReason.TEMPORAL_INSTABILITY,
        ),
    )
    for score, threshold, reason in ordered_gates:
        if score is not None and score >= threshold:
            return _not_assessable(inputs, reason, score)

    probability = float(inputs.breathing_probability)
    breathing = probability >= config.breathing_threshold
    return GateDecision(
        status="breathing" if breathing else "none",
        confidence=_decision_margin(probability, config.breathing_threshold),
        reason=(
            GateReason.BREATHING_EVIDENCE
            if breathing
            else GateReason.NO_BREATHING_EVIDENCE
        ),
        breathing_probability=probability,
        quality_flags=tuple(inputs.quality_flags),
    )
