"""What one frame of this module's replay must contain, besides "t" (seconds on the replay timeline).

This module owns its own fields. Change them here, in the frontend meta.js `summarize`, and in the frontend index.jsx
page, all inside this module's folders. Fusion (frontend/src/features/fusion/rules.js) reads these fields, so tell its owner.
"""

FRAME_FIELDS = ['label', 'confidence', 'probs']

FIELD_NOTES = {
    "label": "Human, Pet, IRobot, Fan or unknown (unknown = highest probability below the threshold; a decision state, not a class)",
    "confidence": "the highest of the four calibrated probabilities, 0 to 1 (0 when the window is invalid)",
    "probs": "calibrated probability of each class, sum 1; null when the window is invalid. Always given, also when the label is unknown",
    "quality": "optional: {ok, level: ok | degraded | invalid, reasons: [...]}; see INTERFACE.md"
}
