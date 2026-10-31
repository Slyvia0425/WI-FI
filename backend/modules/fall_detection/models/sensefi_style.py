"""Independent temporal CNN inspired by SenseFi's shallow encoder/classifier layout.

This is not the original SenseFi Conv2d model and uses no SenseFi weights.
Mixed batches are grouped HP then ESP32; callers must use the same label order.
"""
import torch
from torch import nn


class SenseFiStyleCNN(nn.Module):
    feature_counts = {'HP': 232, 'ESP32': 64}

    def __init__(self, dropout=0.3):
        super().__init__()
        self.adapters = nn.ModuleDict({
            device: nn.Sequential(nn.Conv1d(features, 32, 1), nn.ReLU())
            for device, features in self.feature_counts.items()
        })
        self.encoder = nn.Sequential(
            nn.Conv1d(32, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 96, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(96, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1),
        )

    def forward(self, inputs):
        if not inputs or set(inputs) - set(self.feature_counts):
            raise ValueError('Expected a nonempty device-to-tensor dictionary: HP/ESP32')
        adapted = []
        for device, features in self.feature_counts.items():
            if device not in inputs:
                continue
            x = inputs[device]
            if x.ndim != 3 or x.shape[0] < 1 or tuple(x.shape[1:]) != (features, 500):
                raise ValueError(f'{device}: expected [B,{features},500], got {tuple(x.shape)}')
            if not torch.isfinite(x).all():
                raise ValueError('Input contains NaN/Inf')
            adapted.append(self.adapters[device](x))
        return self.classifier(self.encoder(torch.cat(adapted, dim=0)))


def collate_device_segments(samples):
    """Remove only verified trailing padding; preserve original constant positions.

    Reuses frozen train-fitted normalization from FallDataset. Sorting labels and
    IDs with the tensors makes mixed-device prediction order explicit.
    """
    inputs, ordered = {}, []
    for device, features in SenseFiStyleCNN.feature_counts.items():
        group = [s for s in samples if s['device'] == device]
        if not group:
            continue
        for sample in group:
            mask = sample['feature_mask']
            if not mask[:features].all() or mask[features:].any():
                raise ValueError('Unexpected original-feature mask')
            if torch.count_nonzero(sample['x'][features:]):
                raise ValueError('Nonzero data in expected trailing padding')
        inputs[device] = torch.stack([s['x'][:features] for s in group])
        ordered.extend(group)
    if len(ordered) != len(samples) or not ordered:
        raise ValueError('Unknown device or empty batch')
    return {'x': inputs, 'y': torch.stack([s['y'] for s in ordered]),
            'sample_id': [s['sample_id'] for s in ordered],
            'device': [s['device'] for s in ordered]}
