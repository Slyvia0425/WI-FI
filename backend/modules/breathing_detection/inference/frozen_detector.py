"""Read-only callable adapter from the frozen checkpoint to streaming input."""
import numpy as np
import torch

from ..training.lstm import LightweightBreathingLSTM


class FrozenBreathingDetector:
    def __init__(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if checkpoint.get('representation') != 'interpolated':
            raise ValueError('Expected ordered-interpolation checkpoint')
        self.threshold = float(checkpoint['threshold'])
        self.model = LightweightBreathingLSTM(input_size=int(checkpoint['input_size']),
            hidden_size=int(checkpoint['hidden_size']), projection_size=int(checkpoint['projection_size']))
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()

    @torch.no_grad()
    def __call__(self, sequence):
        return float(torch.sigmoid(self.model(torch.from_numpy(np.asarray(sequence, dtype=np.float32))[None]))[0])
