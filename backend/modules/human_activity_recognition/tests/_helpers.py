from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from ..config import AppConfig, load_config


def config_for(tmp_path: Path, **overrides) -> AppConfig:
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    (data_root / "Multitask" / "HumanActivityRecognition").mkdir(parents=True, exist_ok=True)
    (data_root / "Multitask" / "sub_Human_h5").mkdir(parents=True, exist_ok=True)
    output_root = tmp_path / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    config = load_config(env={"HAR_DATA_ROOT": str(data_root), "HAR_OUTPUT_ROOT": str(output_root)})
    if overrides:
        config = replace(config, **overrides)
    return config


def tiny_config(tmp_path: Path) -> AppConfig:
    config = config_for(tmp_path)
    model = replace(
        config.model,
        epochs=1,
        patience=1,
        batch_size=4,
        hidden_dim=16,
        transformer_heads=4,
        transformer_layers=1,
        max_parameters=2_000_000,
        seeds=(7,),
    )
    runtime = replace(config.runtime, max_batch_size=8, max_concurrent_requests=1, requests_per_second=1000.0)
    return replace(config, model=model, runtime=runtime)


def sample_window(*, feature_size: int = 8, frames: int = 32, seed: int = 1, scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.normal(0.0, scale, size=(feature_size, frames))).astype(np.float32)
