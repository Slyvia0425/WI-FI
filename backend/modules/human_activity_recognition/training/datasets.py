"""In-memory window dataset and controlled augmentation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from ..errors import DataValidationError


@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool = False
    max_shift: int = 8
    amplitude_scale: float = 0.05
    noise_std: float = 0.005


class CSIWindowDataset(Dataset):
    def __init__(
        self,
        windows: Sequence[np.ndarray],
        labels: Sequence[int],
        *,
        transform: object | None = None,
        augmentation: AugmentationConfig | None = None,
        seed: int = 42,
    ) -> None:
        if len(windows) != len(labels):
            raise DataValidationError("windows and labels must have equal length")
        if not windows:
            raise DataValidationError("dataset must not be empty")
        self.windows = [np.asarray(window, dtype=np.float32) for window in windows]
        self.labels = [int(label) for label in labels]
        self.transform = transform
        self.augmentation = augmentation or AugmentationConfig()
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.windows[index]
        if self.transform is not None:
            window = self.transform.transform(window)  # type: ignore[attr-defined]
        if self.augmentation.enabled:
            window = self._augment(window)
        return torch.from_numpy(np.asarray(window, dtype=np.float32)), torch.tensor(
            self.labels[index], dtype=torch.long
        )

    def _augment(self, window: np.ndarray) -> np.ndarray:
        output = window.copy()
        if self.augmentation.max_shift > 0:
            shift = int(self.rng.integers(-self.augmentation.max_shift, self.augmentation.max_shift + 1))
            output = np.roll(output, shift=shift, axis=1)
        if self.augmentation.amplitude_scale > 0:
            output = output * float(
                1.0 + self.rng.uniform(-self.augmentation.amplitude_scale, self.augmentation.amplitude_scale)
            )
        if self.augmentation.noise_std > 0:
            output = output + self.rng.normal(0.0, self.augmentation.noise_std, size=output.shape).astype(np.float32)
        return output
