"""Lightweight bidirectional LSTM for breathing evidence."""

from __future__ import annotations

import torch
from torch import nn


class LightweightBreathingLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 64,
        bidirectional: bool = True,
        dropout: float = 0.2,
        projection_size: int | None = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.projection_size = projection_size
        if projection_size is not None and projection_size < input_size:
            self.input_projection = nn.Sequential(
                nn.Linear(input_size, projection_size),
                nn.LayerNorm(projection_size),
                nn.GELU(),
            )
            lstm_input_size = projection_size
        else:
            self.input_projection = nn.Identity()
            lstm_input_size = input_size
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        output_size = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(output_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected [batch,time,{self.input_size}], got {tuple(x.shape)}"
            )
        sequence, _ = self.lstm(self.input_projection(x))
        pooled = sequence.mean(dim=1)
        return self.head(pooled).squeeze(-1)
