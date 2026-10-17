"""Public training surface for Module B."""

from .datasets import AugmentationConfig, CSIWindowDataset
from .models import ModelSpec, build_model, count_parameters
from .trainer import TrainingResult, TrainingSettings, predict_logits, set_deterministic, train_model

__all__ = [
    "AugmentationConfig",
    "CSIWindowDataset",
    "ModelSpec",
    "TrainingResult",
    "TrainingSettings",
    "build_model",
    "count_parameters",
    "predict_logits",
    "set_deterministic",
    "train_model",
]
