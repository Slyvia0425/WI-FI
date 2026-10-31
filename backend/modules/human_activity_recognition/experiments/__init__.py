"""Public experiment surface for Module B."""

from .runner import (
    SplitBundle,
    WindowDataset,
    load_dataset_split,
    model_spec_from_config,
    run_single_model,
    synthetic_split_bundle,
)

__all__ = [
    "SplitBundle",
    "WindowDataset",
    "load_dataset_split",
    "model_spec_from_config",
    "run_single_model",
    "synthetic_split_bundle",
]
