"""Inspect tensors, scan finite values/padding, and plot train-only signal examples."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

if __package__:
    from .dataset import FallDataset
    from .preprocess import read_amplitude
else:
    from dataset import FallDataset
    from preprocess import read_amplitude


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    before = hashlib.sha256(args.config.read_bytes()).hexdigest()
    train = FallDataset(args.manifest, args.config, 'train_id')
    config = train.preprocessor.config
    assert set(config['training_sample_ids']) == {r['sample_id'] for r in train.rows}
    # Include both labels and both devices deliberately, without changing the official split.
    representatives = []
    for device in ['HP', 'ESP32']:
        for label in ['1', '0']:
            representatives.append(next(i for i, r in enumerate(train.rows)
                                        if r['device'] == device and r['binary_label'] == label))
    indices = representatives + [i for i in range(len(train)) if i not in representatives][:28]
    batch = next(iter(DataLoader(Subset(train, indices), batch_size=32, shuffle=False)))
    examples = []
    for i in representatives:
        sample = train[i]
        examples.append({'sample_id': sample['sample_id'], 'device': sample['device'],
                         'x_shape': list(sample['x'].shape), 'x_dtype': str(sample['x'].dtype),
                         'y': sample['y'].item(), 'y_shape': list(sample['y'].shape),
                         'y_dtype': str(sample['y'].dtype),
                         'present_features': sample['valid_features'],
                         'variable_training_features': int(sample['variable_feature_mask'].sum()),
                         'present_range': [float(sample['x'][sample['feature_mask']].min()),
                                           float(sample['x'][sample['feature_mask']].max())]})
    results = {'examples': examples, 'batch': {
        'x_shape': list(batch['x'].shape), 'y_shape': list(batch['y'].shape),
        'mask_shape': list(batch['feature_mask'].shape), 'labels': batch['y'].tolist(),
        'label_counts': dict(collections.Counter(str(int(y)) for y in batch['y'])),
        'devices': dict(collections.Counter(batch['device'])),
        'range': [float(batch['x'].min()), float(batch['x'].max())]}, 'splits': {},
        'normalization_check': {}, 'plot_examples': []}
    for split in ['train_id', 'val_id', 'test_id']:
        dataset = train if split == 'train_id' else FallDataset(args.manifest, args.config, split)
        total = bad = padded_nonzero = 0
        low, high = float('inf'), -float('inf')
        accum = {}
        for b in DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0):
            x = b['x'].numpy()
            mask = b['feature_mask'].numpy()
            bad += int((~np.isfinite(x)).sum())
            padded_nonzero += int(np.count_nonzero(x[~mask]))
            low = min(low, float(x.min())); high = max(high, float(x.max()))
            total += len(x)
            if split == 'train_id':
                for device in set(b['device']):
                    features = config['devices'][device]['features']
                    z = x[np.array(b['device']) == device, :features].astype(np.float64)
                    state = accum.setdefault(device, {'n': 0, 'sum': np.zeros(features), 'sumsq': np.zeros(features)})
                    state['n'] += z.shape[0] * z.shape[2]
                    state['sum'] += z.sum(axis=(0, 2)); state['sumsq'] += (z*z).sum(axis=(0, 2))
        results['splits'][split] = {'n': total, 'nonfinite_values': bad,
                                   'nonzero_padding_values': padded_nonzero, 'range': [low, high]}
        assert bad == 0 and padded_nonzero == 0
        for device, state in accum.items():
            mean = state['sum']/state['n']
            std = np.sqrt(np.maximum(0, state['sumsq']/state['n'] - mean**2))
            variable = np.ones(len(mean), dtype=bool)
            variable[config['devices'][device]['constant_training_features']] = False
            assert np.max(np.abs(mean)) < 1e-5
            assert np.max(np.abs(std[variable]-1)) < 1e-5
            assert state['n'] == config['devices'][device]['observations_per_feature']
            results['normalization_check'][device] = {'observations_per_original_feature': state['n'],
                'max_abs_mean': float(np.max(np.abs(mean))),
                'max_abs_std_minus_one_nonconstant': float(np.max(np.abs(std[variable]-1))),
                'constant_training_features': config['devices'][device]['constant_training_features']}
        print(split, results['splits'][split], flush=True)
    # Show all four device/class combinations; use original, nonconstant positions only.
    fig, axes = plt.subplots(4, 2, figsize=(13, 12), sharex=True, layout='constrained')
    for n, i in enumerate(representatives):
        row, sample = train.rows[i], train[i]
        raw = read_amplitude(row['file_path'])
        candidates = np.flatnonzero(sample['variable_feature_mask'].numpy()[:len(raw)] & (raw.std(axis=1) > 1e-8))
        if len(candidates) < 3:
            raise ValueError('Not enough variable original features to plot')
        selected = candidates[np.linspace(0, len(candidates)-1, 3, dtype=int)]
        for feature in selected:
            axes[n, 0].plot(np.arange(raw.shape[1]), raw[feature], lw=.8, label=f'feature {feature}')
            axes[n, 1].plot(np.arange(raw.shape[1]), sample['x'][feature], lw=.8, label=f'feature {feature}')
        label = 'Fall' if sample['y'].item() == 1 else 'Nonfall'
        axes[n, 0].set_title(f"{row['device']} / {label} — original amplitude")
        axes[n, 1].set_title(f"{row['device']} / {label} — train-stat normalized")
        axes[n, 0].set_ylabel('Amplitude'); axes[n, 1].set_ylabel('z-score')
        for ax in axes[n]:
            ax.legend(fontsize=8); ax.grid(alpha=.2)
        results['plot_examples'].append({'sample_id': row['sample_id'], 'features': selected.tolist(),
                                         'selection': 'first/middle/last among nonconstant original feature positions'})
    for ax in axes[-1]:
        ax.set_xlabel('Time sample index (sampling rate unverified)')
    fig.suptitle('Complete training segments: reading check, not evidence of classification performance')
    fig.savefig(args.output_dir/'signal_examples.png', dpi=150)
    plt.close(fig)
    after = hashlib.sha256(args.config.read_bytes()).hexdigest()
    assert before == after
    results['preprocess_config_unchanged_sha256'] = after
    results['versions'] = {'torch': torch.__version__, 'numpy': np.__version__, 'matplotlib': matplotlib.__version__}
    (args.output_dir/'dataset_check.json').write_text(json.dumps(results, indent=2))
    print(json.dumps({'examples': examples, 'batch': results['batch'],
                      'normalization_check': results['normalization_check']}, indent=2))


if __name__ == '__main__':
    main()
