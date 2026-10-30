"""Training-reference diagnostics for ordered-frequency and embedding shift."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from backend.modules.breathing_detection.preprocessing import WireBreathDataset
from backend.modules.breathing_detection.preprocessing.cache import load_temporal_cache
from backend.modules.breathing_detection.preprocessing.wire_cache import load_wire_breath_cache
from backend.modules.breathing_detection.training.lstm import LightweightBreathingLSTM
from .analyze_wire_breath_failures import TARGET_RECORDING
from .evaluate_lstm import checkpoint_sha256


def adjacent_correlations(values):
    centered = values - values.mean(axis=0, keepdims=True)
    norms = np.sqrt(np.sum(centered ** 2, axis=0))
    return np.sum(centered[:, :-1] * centered[:, 1:], axis=0) / np.maximum(
        norms[:-1] * norms[1:], 1e-12
    )


@torch.no_grad()
def representations(model, sequences, indices):
    projected, pooled, probabilities, correlations = [], [], [], []
    for start in range(0, len(indices), 64):
        batch = np.asarray(sequences[indices[start:start + 64]], dtype=np.float32)
        x = torch.from_numpy(batch)
        z = model.input_projection(x)
        h, _ = model.lstm(z)
        embedding = h.mean(dim=1)
        projected.append(z.mean(dim=1).numpy())
        pooled.append(embedding.numpy())
        probabilities.append(torch.sigmoid(model.head(embedding)).numpy().reshape(-1))
        correlations.extend(float(np.median(adjacent_correlations(row))) for row in batch)
    return np.concatenate(projected), np.concatenate(pooled), np.concatenate(probabilities), np.asarray(correlations)


def distances(reference, queries):
    # Shrinkage keeps the metric stable for correlated neural coordinates.
    mean = reference.mean(axis=0)
    covariance = np.cov(reference, rowvar=False)
    covariance = 0.9 * covariance + 0.1 * np.eye(len(mean)) * np.trace(covariance) / len(mean)
    inverse = np.linalg.pinv(covariance)
    delta = queries - mean
    return np.sqrt(np.maximum(np.einsum('ni,ij,nj->n', delta, inverse, delta), 0))


def summary(values):
    return {"median": float(np.median(values)), "p10": float(np.quantile(values, .1)), "p90": float(np.quantile(values, .9))}


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
    train = load_temporal_cache(base / 'temporal_cache', 'train_id', 70000)
    wire = load_wire_breath_cache(base / 'wire_breath_cache', 1401)
    rng = np.random.default_rng(20261028)
    # Independent fitting and calibration subsets, exclusively from training.
    indices = np.concatenate([
        rng.choice(np.flatnonzero((train.labels == label) & (train.weights > 0)), 1200, replace=False)
        for label in (0, 1)
    ])
    print('Extracting 2,400 training and 1,401 external representations', flush=True)
    tp, th, _, tc = representations(model, train.sequences, indices)
    wp, wh, probabilities, wc = representations(model, wire.sequences, np.arange(1401))
    labels = np.asarray(train.labels[indices])
    metrics = {}
    for name, training, external in [('projection', tp, wp), ('lstm', th, wh)]:
        for label in (0, 1):
            reference = training[labels == label]
            calibration = distances(reference[:800], reference[800:])
            distance = distances(reference[:800], external)
            metrics[f'{name}_to_{"empty" if label == 0 else "breathing"}'] = (
                distance, calibration
            )
    samples = wire.manifest['samples']
    target = np.asarray([s['recording_id'] == TARGET_RECORDING for s in samples])
    valid = np.asarray(wire.weights, dtype=bool)
    detected = probabilities >= checkpoint['threshold']
    groups = {
        'target_failed': target & ~detected & valid,
        'target_successful': target & detected & valid,
        'subject02_sitting_interference': np.asarray([s['participant_id'] == '02' and s['posture'] == 'Sitting' and s['environment'] == 'Interference' for s in samples]) & valid,
        'other_sitting_successful': np.asarray([s['posture'] == 'Sitting' for s in samples]) & ~target & detected & valid,
        'all_other_successful': ~target & detected & valid,
    }
    # Preserve original frequency indices; do not compare raw amplitudes across devices.
    raw = WireBreathDataset('data/WiReBreath', normalize=False)
    profiles = {name: {'mean': [], 'variance': [], 'adjacent_correlation': []} for name in groups}
    for i in range(len(raw)):
        selected = [name for name, mask in groups.items() if mask[i]]
        if not selected:
            continue
        x = raw[i].x[0].T.astype(np.float64)
        for name in selected:
            profiles[name]['mean'].append(x.mean(axis=0))
            profiles[name]['variance'].append(x.var(axis=0))
            profiles[name]['adjacent_correlation'].append(adjacent_correlations(x))
    results = {}
    for name, mask in groups.items():
        result = {'windows': int(mask.sum()), 'model_probability': summary(probabilities[mask]), 'model_input_adjacent_correlation': summary(wc[mask])}
        for metric, (distance, calibration) in metrics.items():
            result[metric] = summary(distance[mask])
            result[metric]['above_training_calibration_p95_fraction'] = float(np.mean(distance[mask] > np.quantile(calibration, .95)))
        result['raw_profiles'] = {key: np.median(values, axis=0).tolist() for key, values in profiles[name].items()}
        result['raw_adjacent_correlation'] = summary(np.median(profiles[name]['adjacent_correlation'], axis=1))
        results[name] = result
    report = {
        'checkpoint_sha256': checkpoint_sha256(path),
        'seed': 20261028,
        'reference': '1,200 per class from CSI-Bench train only; 800 fit + 400 calibration',
        'distance': 'shrinkage Mahalanobis; projection temporal mean and LSTM temporal mean',
        'training_model_input_adjacent_correlation': {str(label): summary(tc[labels == label]) for label in (0, 1)},
        'calibration_distances': {name: summary(calibration) for name, (_, calibration) in metrics.items()},
        'groups': results,
        'target_profile_comparisons': {
            name: {
                field: {
                    'log1p_profile_correlation': float(np.corrcoef(
                        np.log1p(results['target_failed']['raw_profiles'][field]),
                        np.log1p(result['raw_profiles'][field]),
                    )[0, 1]),
                }
                for field in ('mean', 'variance')
            }
            for name, result in results.items() if name != 'target_failed'
        },
        'limitations': ['Normalized frequency position is not proven equivalent absolute RF frequency across devices.', 'Final LSTM embedding distances are influenced by the trained binary decision; they do not establish causation.', 'Windows within one recording are correlated; window counts are not independent replications.'],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({name: {k: v for k, v in result.items() if k != 'raw_profiles'} for name, result in results.items()}, indent=2))


if __name__ == '__main__':
    main()
