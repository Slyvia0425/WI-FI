"""What one frame of this module's replay must contain.

Required fields stay backward compatible for the website and fusion layer. Optional evidence fields are additive and
must never be used as a substitute for a missing required field.
"""

FRAME_FIELDS = ["activity", "confidence", "stable"]

OPTIONAL_FRAME_FIELDS = ["activity_id", "probabilities", "stability", "quality", "model_version", "transition"]

FIELD_NOTES = {
    "activity": "stable activity id: standing, sitting, walking or lying; decision state: transition or unknown",
    "confidence": "probability of the selected decision, 0 to 1",
    "stable": "true only for a stable, sufficiently confident state",
    "activity_id": "optional stable class id 0..3 for stable states, null for decision states",
    "probabilities": "optional mapping of stable ids to probabilities",
    "stability": "optional 0..1 temporal continuity score",
    "quality": "optional input quality: good, degraded or invalid",
    "model_version": "optional immutable model identifier",
    "transition": "optional true when the frame is a transition decision",
}


def validate_frame_values(payload: dict) -> list[str]:
    """Validate value domains that the generic repository contract cannot know."""
    problems: list[str] = []
    allowed = {"standing", "sitting", "walking", "lying", "transition", "unknown"}
    for index, frame in enumerate(payload.get("frames", [])):
        activity = frame.get("activity")
        if activity not in allowed:
            problems.append(f"frame {index}: unsupported activity {activity!r}")
        confidence = frame.get("confidence")
        if not isinstance(confidence, int | float) or not 0 <= float(confidence) <= 1:
            problems.append(f"frame {index}: confidence must be within [0, 1]")
        if not isinstance(frame.get("stable"), bool):
            problems.append(f"frame {index}: stable must be boolean")
        probabilities = frame.get("probabilities")
        if probabilities is not None:
            if not isinstance(probabilities, dict):
                problems.append(f"frame {index}: probabilities must be an object")
            else:
                values = list(probabilities.values())
                if any(not isinstance(value, int | float) or not 0 <= float(value) <= 1 for value in values):
                    problems.append(f"frame {index}: probabilities must be within [0, 1]")
                elif abs(sum(float(value) for value in values) - 1.0) > 1e-3:
                    problems.append(f"frame {index}: probabilities must sum to 1")
        quality = frame.get("quality")
        if quality is not None and quality not in {"good", "degraded", "invalid"}:
            problems.append(f"frame {index}: unsupported quality {quality!r}")
    return problems
