"""Inference and decision gating for breathing detection."""

from .gating import (
    GateConfig,
    GateDecision,
    GateInputs,
    GateReason,
    decide_breathing_status,
)
from .periodic_risk import PeriodicInterferenceScorer
from .temporal_risk import TemporalInstabilityTracker
from .streaming import BreathingStream, StreamConfig

__all__ = [
    "GateConfig",
    "GateDecision",
    "GateInputs",
    "GateReason",
    "decide_breathing_status",
    "PeriodicInterferenceScorer",
    "TemporalInstabilityTracker",
    "BreathingStream",
    "StreamConfig",
]
