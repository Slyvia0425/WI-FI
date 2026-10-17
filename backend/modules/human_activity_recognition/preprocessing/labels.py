"""Stable label mapping. Display text is never used as an identifier."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..domain import ActivityState
from ..errors import DataValidationError, InputValidationError

_ALIASES = {
    "stand": ActivityState.STANDING,
    "standing": ActivityState.STANDING,
    "stood": ActivityState.STANDING,
    "sit": ActivityState.SITTING,
    "sitting": ActivityState.SITTING,
    "seated": ActivityState.SITTING,
    "walk": ActivityState.WALKING,
    "walking": ActivityState.WALKING,
    "lie": ActivityState.LYING,
    "lying": ActivityState.LYING,
    "lyingdown": ActivityState.LYING,
    "reclining": ActivityState.LYING,
    "transition": ActivityState.TRANSITION,
    "unknown": ActivityState.UNKNOWN,
}


def _normalise_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.strip().lower())


def canonical_activity(value: str, *, allow_decision_state: bool = True) -> ActivityState:
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError("activity label must be a non-empty string")
    state = _ALIASES.get(_normalise_key(value))
    if state is None:
        raise DataValidationError("activity label is not supported", {"label": value})
    if not allow_decision_state and state in {ActivityState.TRANSITION, ActivityState.UNKNOWN}:
        raise DataValidationError("decision state is not a training class", {"label": value})
    return state


def read_label_mapping(path: Path) -> dict[str, ActivityState]:
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DataValidationError("label mapping file not found", {"path": str(path)}) from exc
    except json.JSONDecodeError as exc:
        raise DataValidationError("label mapping is not valid JSON", {"path": str(path)}) from exc
    if isinstance(raw, list):
        pairs = enumerate(raw)
        mapping = {str(key): value for key, value in pairs}
    elif isinstance(raw, Mapping):
        if all(isinstance(value, str) for value in raw.values()):
            mapping = {str(key): value for key, value in raw.items()}
        else:
            mapping = {str(value): key for key, value in raw.items()}
    else:
        raise DataValidationError("label mapping must be a list or mapping", {"path": str(path)})
    return {str(key): canonical_activity(str(value)) for key, value in mapping.items()}


def activity_id(state: ActivityState) -> str:
    return state.value
