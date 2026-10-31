"""Independent full-segment CNN+LSTM variants; no pretrained weights or state carry."""
import torch
from torch import nn
from .cnn import FallCNN
from .sensefi_style import SenseFiStyleCNN


class TimeSequence(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2).contiguous()


class LastHidden(nn.Module):
    def forward(self, result):
        _, (hidden, _) = result
        return hidden[-1]


def sequence_head(features, dropout):
    return nn.Sequential(TimeSequence(), nn.LSTM(features, 64, batch_first=True),
                         LastHidden(), nn.Dropout(dropout), nn.Linear(64, 1))


class OriginalCNNLSTM(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.encoder = FallCNN().layers[:6]
        self.head = sequence_head(64, dropout)

    def forward(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (232, 500) or not torch.isfinite(x).all():
            raise ValueError('Expected finite [B,232,500] input')
        return self.head(self.encoder(x))


class SenseFiStyleCNNLSTM(nn.Module):
    feature_counts = {'HP': 232, 'ESP32': 64}

    def __init__(self, dropout=0.3):
        super().__init__()
        base = SenseFiStyleCNN()
        self.adapters = base.adapters
        self.encoder = base.encoder[:8]
        self.head = sequence_head(96, dropout)

    def forward(self, inputs):
        if not inputs or set(inputs)-set(self.feature_counts):
            raise ValueError('Expected nonempty HP/ESP32 input dictionary')
        parts = []
        for device, features in self.feature_counts.items():
            if device not in inputs:
                continue
            x = inputs[device]
            if x.ndim != 3 or x.shape[0] < 1 or tuple(x.shape[1:]) != (features, 500) or not torch.isfinite(x).all():
                raise ValueError(f'{device}: expected finite [B,{features},500]')
            parts.append(self.adapters[device](x))
        return self.head(self.encoder(torch.cat(parts, dim=0)))
