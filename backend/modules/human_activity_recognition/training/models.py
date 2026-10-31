"""1D-CNN and CNN-Transformer models for five-class activity recognition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ..errors import ModelValidationError


class HAR1DCNN(nn.Module):
    def __init__(self, *, feature_size: int, num_classes: int, hidden_dim: int = 128, dropout: float = 0.15) -> None:
        super().__init__()
        channels = max(16, hidden_dim // 4)
        self.encoder = nn.Sequential(
            nn.Conv1d(feature_size, channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(channels, channels * 2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(channels * 2, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(inputs))


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_length: int = 4096) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(1, max_length, dim))
        nn.init.normal_(self.embedding, std=0.02)
        self.max_length = max_length

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        length = inputs.shape[1]
        if length > self.max_length:
            raise ModelValidationError(
                "sequence exceeds positional encoding capacity", {"length": length, "capacity": self.max_length}
            )
        return inputs + self.embedding[:, :length, :]


class HARCNNTransformer(nn.Module):
    def __init__(
        self,
        *,
        feature_size: int,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.15,
        heads: int = 4,
        layers: int = 2,
    ) -> None:
        super().__init__()
        conv_channels = max(16, hidden_dim // 2)
        self.conv = nn.Sequential(
            nn.Conv1d(feature_size, conv_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(conv_channels, hidden_dim, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.position = PositionalEncoding(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.conv(inputs).transpose(1, 2)
        encoded = self.position(encoded)
        encoded = self.transformer(encoded)
        return self.classifier(encoded.mean(dim=1))


@dataclass(frozen=True)
class ModelSpec:
    name: str
    feature_size: int
    num_classes: int
    hidden_dim: int
    dropout: float
    heads: int
    layers: int


def build_model(spec: ModelSpec) -> nn.Module:
    if spec.name == "cnn_1d":
        return HAR1DCNN(
            feature_size=spec.feature_size,
            num_classes=spec.num_classes,
            hidden_dim=spec.hidden_dim,
            dropout=spec.dropout,
        )
    if spec.name == "cnn_transformer":
        if spec.hidden_dim % spec.heads:
            raise ModelValidationError(
                "hidden_dim must be divisible by transformer heads",
                {"hidden_dim": spec.hidden_dim, "heads": spec.heads},
            )
        return HARCNNTransformer(
            feature_size=spec.feature_size,
            num_classes=spec.num_classes,
            hidden_dim=spec.hidden_dim,
            dropout=spec.dropout,
            heads=spec.heads,
            layers=spec.layers,
        )
    raise ModelValidationError("unknown model name", {"name": spec.name})


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def assert_parameter_budget(model: nn.Module, maximum: int) -> int:
    count = count_parameters(model)
    if count > maximum:
        raise ModelValidationError("model exceeds parameter budget", {"parameters": count, "maximum": maximum})
    return count


def model_metadata(spec: ModelSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "feature_size": spec.feature_size,
        "num_classes": spec.num_classes,
        "hidden_dim": spec.hidden_dim,
        "dropout": spec.dropout,
        "heads": spec.heads,
        "layers": spec.layers,
    }
