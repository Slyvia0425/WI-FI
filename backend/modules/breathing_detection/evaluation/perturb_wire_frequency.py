"""Controlled frequency-order sensitivity diagnostics for the frozen LSTM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from backend.modules.breathing_detection.preprocessing.cache import load_temporal_cache
from backend.modules.breathing_detection.preprocessing.wire_cache import load_wire_breath_cache
from backend.modules.breathing_detection.training.lstm import LightweightBreathingLSTM
from .analyze_wire_breath_failures import TARGET_RECORDING
from .evaluate_lstm import checkpoint_sha256


def perturb(values: np.ndarray, shift: int = 0, reverse: bool = False, edge: bool = False) -> np.ndarray:
    """Circularly shift the frequency axis; preserve every temporal stream."""
    result = values[..., ::-1] if reverse else values
    shifted = np.roll(result, shift, axis=-1)
    if edge and shift > 0:
        shifted[..., :shift] = result[..., :1]
    elif edge and shift < 0:
        shifted[..., shift:] = result[..., -1:]
    return np.ascontiguousarray(shifted)


@torch.no_grad()
def score(model, sequences, shift=0, reverse=False, edge=False):
    result = []
    for start in range(0, len(sequences), 64):
        batch = perturb(np.asarray(sequences[start:start + 64], dtype=np.float32), shift, reverse, edge)
        result.append(torch.sigmoid(model(torch.from_numpy(batch))).numpy())
    return np.concatenate(result)


def changes(probabilities, baseline, mask, threshold):
    detected = probabilities[mask] >= threshold
    previous = baseline[mask] >= threshold
    return {
        'count': int(mask.sum()),
        'breathing_rate': float(detected.mean()),
        'recovered': int(np.sum(detected & ~previous)),
        'lost': int(np.sum(~detected & previous)),
        'probability_median': float(np.median(probabilities[mask])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = Path('backend/modules/breathing_detection/outputs')
    path = base / 'lstm_interpolated_full_seed123/best_model.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = LightweightBreathingLSTM(input_size=232, hidden_size=64, projection_size=32)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    threshold = float(checkpoint['threshold'])
    wire = load_wire_breath_cache(base / 'wire_breath_cache', 1401)
    train = load_temporal_cache(base / 'temporal_cache', 'train_id', 70000)
    rng = np.random.default_rng(20261028)
    indices = np.concatenate([
        rng.choice(np.flatnonzero((train.labels == label) & (train.weights > 0)), 1200, replace=False)
        for label in (0, 1)
    ])
    training_sequences = np.asarray(train.sequences[indices], dtype=np.float32)
    labels = np.asarray(train.labels[indices])
    baseline = score(model, wire.sequences)
    training_baseline = score(model, training_sequences)
    samples = wire.manifest['samples']
    valid = np.asarray(wire.weights, dtype=bool)
    target = np.asarray([s['recording_id'] == TARGET_RECORDING for s in samples])
    groups = {
        'target_recording': target & valid,
        'target_baseline_failed': target & valid & (baseline < threshold),
        'same_subject_sitting_interference': valid & np.asarray([
            s['participant_id'] == '02' and s['posture'] == 'Sitting' and s['environment'] == 'Interference'
            for s in samples
        ]),
        'other_baseline_successful': valid & ~target & (baseline >= threshold),
        'all_assessable': valid,
    }
    rows = []
    configurations = [('identity', 0, False, False)] + [
        (f'circular_shift_{shift:+d}', shift, False, False) for shift in (-32, -16, -4, -1, 1, 4, 16, 32)
    ] + [('reverse', 0, True, False)] + [
        (f'edge_shift_{shift:+d}', shift, False, True) for shift in (-4, -1, 1, 4)
    ]
    for name, shift, reverse, edge in configurations:
        probabilities = baseline if name == 'identity' else score(model, wire.sequences, shift, reverse, edge)
        training_probabilities = training_baseline if name == 'identity' else score(model, training_sequences, shift, reverse, edge)
        row = {
            'transform': name,
            'wire': {key: changes(probabilities, baseline, mask, threshold) for key, mask in groups.items()},
            'training_reference': {
                'empty': changes(training_probabilities, training_baseline, labels == 0, threshold),
                'breathing': changes(training_probabilities, training_baseline, labels == 1, threshold),
            },
        }
        rows.append(row)
        print(name, 'target detected', row['wire']['target_recording']['breathing_rate'],
              'other lost', row['wire']['other_baseline_successful']['lost'],
              'train FPR', row['training_reference']['empty']['breathing_rate'], flush=True)
    report = {
        'checkpoint_sha256': checkpoint_sha256(path),
        'threshold': threshold,
        'training_reference': 'seed 20261028; 1,200 per class; training data only; descriptive in-sample controls',
        'method': 'Fixed circular shifts, edge-replicated shifts, and reversal on final 232-position input; time axis preserved',
        'limitations': ['Circular wrap introduces a nonphysical frequency boundary.', 'This is input sensitivity, not proof of physical frequency misalignment.', 'No transform selected for deployment; failed record already observed; positive-only external data cannot assess specificity.'],
        'results': rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
