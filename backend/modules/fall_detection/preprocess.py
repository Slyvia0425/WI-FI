"""Fit train-only device/feature statistics and adapt complete CSI segments."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def manifest_rows(path):
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Manifest is empty or sample IDs are duplicated')
    for row in rows:
        if row['status'] != 'ok' or float(row['finite_ratio']) != 1:
            raise ValueError(f"Audit failure: {row['sample_id']}")
        expected = {'Fall': '1', 'Nonfall': '0'}.get(row['original_label'])
        if expected is None or row['binary_label'] != expected:
            raise ValueError(f"Unverified label: {row['sample_id']}")
    return rows


def read_amplitude(path, key='CSI_amps'):
    with h5py.File(path, 'r') as handle:
        data = handle[key][()]
    if data.ndim != 3 or data.shape[1:] != (500, 1) or data.shape[0] not in (64, 232):
        raise ValueError(f'{path}: unsupported shape {data.shape}; no reshape/resampling permitted')
    if not np.issubdtype(data.dtype, np.floating) or not np.isfinite(data).all():
        raise ValueError(f'{path}: expected finite floating-point amplitude')
    if np.any(data < 0):
        raise ValueError(f'{path}: negative amplitude needs review')
    return data[:, :, 0].astype(np.float64)  # only remove the verified singleton dimension


def fit_normalization(manifest, output):
    rows = manifest_rows(manifest)
    train = [r for r in rows if 'train_id' in r['split'].split(';')]
    if not train:
        raise ValueError('No official train_id records')
    stats = {}
    for i, row in enumerate(train, 1):
        x = read_amplitude(row['file_path'], row['h5_key'])
        device = row['device']
        if device not in stats:
            stats[device] = {'n': 0, 'mean': np.zeros(x.shape[0]), 'm2': np.zeros(x.shape[0]), 'samples': 0}
        state = stats[device]
        if len(state['mean']) != len(x):
            raise ValueError('Feature count changed within device; review input semantics')
        count = x.shape[1]
        mean = x.mean(axis=1)
        m2 = ((x - mean[:, None]) ** 2).sum(axis=1)
        delta = mean - state['mean']
        total = state['n'] + count
        state['m2'] += m2 + delta**2 * state['n'] * count / total
        state['mean'] += delta * count / total
        state['n'] = total
        state['samples'] += 1
        if i % 1000 == 0:
            print(f'Fitted {i}/{len(train)} train segments', flush=True)
    config = {
        'version': 'module_c_preprocess_v1', 'source_layout': ['frequency_feature', 'time', 'singleton'],
        'target_features': 232, 'time_points': 500, 'h5_key': 'CSI_amps',
        'mode': 'per_device_per_feature_train_population_zscore_then_right_zero_pad',
        'physical_link_semantics': 'unknown; singleton removed only',
        'sample_rate_hz': None, 'time_resampling': False, 'windowing': False,
        'epsilon': 1e-8, 'label_mapping': {'Fall': 1, 'Nonfall': 0},
        'training_split': 'train_id', 'train_samples': len(train),
        'training_sample_ids': sorted(r['sample_id'] for r in train),
        'manifest_sha256': hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
        'duplicate_policy': 'retain_official_splits_and_train_duplicates; disclose_audit_risks',
        'padding_in_statistics': False, 'devices': {},
    }
    for device, state in stats.items():
        std = np.sqrt(state['m2'] / state['n'])
        constant = std < config['epsilon']
        config['devices'][device] = {
            'features': len(std), 'train_samples': state['samples'],
            'observations_per_feature': state['n'], 'mean': state['mean'].tolist(),
            'std': std.tolist(), 'scale': np.where(constant, 1., std).tolist(),
            'constant_training_features': np.flatnonzero(constant).tolist(),
        }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(config, indent=2))
    return config


class Preprocessor:
    def __init__(self, config):
        self.config = json.loads(Path(config).read_text()) if isinstance(config, (str, Path)) else config
        if self.config['version'] != 'module_c_preprocess_v1':
            raise ValueError('Unsupported preprocess configuration')

    def transform_file(self, path, device):
        if device not in self.config['devices']:
            raise ValueError(f'Unknown device {device}; no fitted training statistics')
        x = read_amplitude(path, self.config['h5_key'])
        state = self.config['devices'][device]
        if x.shape != (state['features'], self.config['time_points']):
            raise ValueError('Input shape differs from saved device configuration')
        z = (x - np.array(state['mean'])[:, None]) / np.array(state['scale'])[:, None]
        output = np.zeros((self.config['target_features'], self.config['time_points']), dtype=np.float32)
        output[:len(x)] = z
        mask = np.arange(len(output)) < len(x)
        if not np.isfinite(output).all():
            raise ValueError('Nonfinite normalized output')
        return output, mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = fit_normalization(args.manifest, args.output)
    print(json.dumps({'train_samples': config['train_samples'], 'devices': {
        d: {k: v for k, v in s.items() if k not in ('mean', 'std', 'scale')}
        for d, s in config['devices'].items()}}, indent=2))


if __name__ == '__main__':
    main()
