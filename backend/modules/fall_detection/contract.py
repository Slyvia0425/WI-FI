"""What one frame of this module's replay must contain, besides "t" (seconds on the replay timeline).

This module owns its own fields. Change them here, in the frontend meta.js `summarize`, and in the frontend index.jsx
page, all inside this module's folders. Fusion (frontend/src/features/fusion/rules.js) reads these fields, so tell its owner.
"""

FRAME_FIELDS = ['fall_score', 'state']

FIELD_NOTES = {
    "fall_score": "0 to 1 when C ran; null when not evaluated by confidence gate",
    "state": "normal, suspected, alert or not_evaluated (no C binary decision)"
}
