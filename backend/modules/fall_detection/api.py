"""Backend entry point for fall detection. backend/api/routes.py calls replay() for GET /api/fall_detection/replay.

Default: return the file written by evaluation code through backend.shared.export.write_replay.
Replace it with live inference if this module should compute results on request.
"""
from backend.shared.export import read_replay
from .activity_reference import attach_activity_reference

MODULE = "fall_detection"


def replay() -> dict | None:
    payload=read_replay(MODULE)
    if payload and payload.get('mode')=='independent_segment_validation':
        try:
            activity=read_replay('human_activity_recognition')
        except (OSError,ValueError,TypeError):
            result=attach_activity_reference(payload,None)
            result['activity_reference_summary']['source_status']='Module B replay unreadable; C remains available'
            return result
        return attach_activity_reference(payload,activity)
    return payload
