"""Deterministic training loop with validation-based model selection and checkpoint metadata."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..errors import ModelValidationError
from .datasets import AugmentationConfig, CSIWindowDataset
from .models import ModelSpec, assert_parameter_budget, build_model, model_metadata


@dataclass(frozen=True)
class TrainingSettings:
    model: ModelSpec
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    patience: int
    max_parameters: int
    seed: int = 42
    device: str = "cpu"
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass
class TrainingResult:
    model: nn.Module
    best_epoch: int
    best_validation_loss: float
    history: list[dict[str, float]]
    parameter_count: int
    metadata: dict[str, Any]


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def class_weights(labels: Sequence[int], num_classes: int) -> torch.Tensor:
    counts: np.ndarray = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_model(
    *,
    train_windows: Sequence[np.ndarray],
    train_labels: Sequence[int],
    validation_windows: Sequence[np.ndarray],
    validation_labels: Sequence[int],
    settings: TrainingSettings,
    transform: object | None = None,
    checkpoint_path: Path | None = None,
    model_version: str = "development",
    feature_stats: dict[str, object] | None = None,
) -> TrainingResult:
    set_deterministic(settings.seed)
    if not train_windows or not validation_windows:
        raise ModelValidationError("training and validation data must both be non-empty")
    if set(train_labels) != set(validation_labels):
        # Not an error, but the warning is made explicit in metadata.
        missing = sorted(set(train_labels) - set(validation_labels))
    else:
        missing = []
    model = build_model(settings.model)
    parameters = assert_parameter_budget(model, settings.max_parameters)
    device = torch.device(settings.device if settings.device == "cpu" or torch.cuda.is_available() else "cpu")
    model.to(device)
    train_loader = DataLoader(
        CSIWindowDataset(
            train_windows, train_labels, transform=transform, augmentation=settings.augmentation, seed=settings.seed
        ),
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(settings.seed),
    )
    validation_loader = DataLoader(
        CSIWindowDataset(validation_windows, validation_labels, transform=transform),
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_labels, settings.model.num_classes).to(device))
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    stale = 0
    for epoch in range(1, settings.epochs + 1):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += float(loss.detach().cpu()) * inputs.shape[0]
        train_loss /= len(train_loader.dataset)
        validation_loss = _evaluate_loss(model, validation_loader, criterion, device)
        history.append({"epoch": float(epoch), "train_loss": train_loss, "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= settings.patience:
                break
    if best_state is None:
        raise ModelValidationError("training produced no checkpoint")
    model.load_state_dict(best_state)
    metadata = {
        "model_version": model_version,
        "model": model_metadata(settings.model),
        "parameter_count": parameters,
        "seed": settings.seed,
        "validation_labels_missing_from_train": missing,
        "training": asdict(settings),
        "feature_stats": feature_stats,
    }
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "metadata": metadata}, checkpoint_path)
    return TrainingResult(
        model=model,
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        history=history,
        parameter_count=parameters,
        metadata=metadata,
    )


@torch.no_grad()
def _evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        total += float(criterion(model(inputs), targets).detach().cpu()) * inputs.shape[0]
    return total / len(loader.dataset)


@torch.no_grad()
def predict_logits(
    model: nn.Module, windows: Sequence[np.ndarray], *, batch_size: int = 64, device: str = "cpu"
) -> np.ndarray:
    model.eval()
    selected = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    model.to(selected)
    dataset = CSIWindowDataset(windows, [0] * len(windows))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    chunks: list[np.ndarray] = []
    for inputs, _ in loader:
        chunks.append(model(inputs.to(selected)).detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)
