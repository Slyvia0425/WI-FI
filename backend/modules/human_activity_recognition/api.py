"""Backend entry point for human activity recognition.

backend/api/routes.py calls replay() for GET /api/human_activity_recognition/replay.

Default: return the file written by evaluation code through backend.shared.export.write_replay.
Replace it with live inference if this module should compute results on request.
"""

from backend.shared.export import read_replay

MODULE = "human_activity_recognition"


def replay() -> dict | None:
    return read_replay(MODULE)
