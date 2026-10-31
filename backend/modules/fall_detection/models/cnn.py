"""Minimal complete-segment 1D CNN; returns one raw logit per segment."""
import torch
from torch import nn


class FallCNN(nn.Module):
    def __init__(self, in_features=232, dropout=0.3):
        super().__init__()
        self.config = {'in_features': in_features, 'dropout': dropout}
        self.layers = nn.Sequential(
            nn.Conv1d(in_features, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != self.config['in_features'] or x.shape[2] < 4:
            raise ValueError(f'Expected [B,{self.config["in_features"]},T>=4], got {tuple(x.shape)}')
        return self.layers(x)
