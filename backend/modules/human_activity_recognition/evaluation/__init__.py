"""Public evaluation surface for Module B."""

from .error_analysis import class_support, top_confusions
from .metrics import ClassificationMetrics, classification_metrics, confusion_matrix, expected_calibration_error
from .replay import ReplayArtifacts, export_artifacts, frames_from_predictions, make_metrics_block

__all__ = [
    "ClassificationMetrics",
    "ReplayArtifacts",
    "class_support",
    "classification_metrics",
    "confusion_matrix",
    "expected_calibration_error",
    "export_artifacts",
    "frames_from_predictions",
    "make_metrics_block",
    "top_confusions",
]
