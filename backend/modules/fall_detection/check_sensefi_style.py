"""Training pipeline diagnostics only: 2-epoch smoke and fixed-subset overfit.

Only official train_id samples are loaded. No validation, test, model selection,
production checkpoint, or full-data training is performed.
"""
import argparse
import collections
import csv
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

if __package__:
    from .dataset import FallDataset
    from .models.sensefi_style import SenseFiStyleCNN, collate_device_segments
else:
    from dataset import FallDataset
    from models.sensefi_style import SenseFiStyleCNN, collate_device_segments


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def pick_subset(dataset, per_group, seed):
    # Unique content within the diagnostic subset. Keep the official split untouched.
    groups = collections.defaultdict(list)
    for i, row in enumerate(dataset.rows):
        groups[(row['device'], row['binary_label'])].append(i)
    rng = random.Random(seed)
    indices, seen = [], set()
    for group, candidates in sorted(groups.items()):
        rng.shuffle(candidates)
        chosen = 0
        for i in candidates:
            content_hash = dataset.rows[i]['content_hash']
            if content_hash in seen:
                continue
            indices.append(i); seen.add(content_hash); chosen += 1
            if chosen == per_group:
                break
        if chosen != per_group:
            raise ValueError(f'Insufficient distinct training examples: {group}')
    return indices


def describe_subset(dataset, indices):
    return [{'sample_id': dataset.rows[i]['sample_id'], 'device': dataset.rows[i]['device'],
             'label': int(dataset.rows[i]['binary_label']), 'content_hash': dataset.rows[i]['content_hash']}
            for i in indices]


def evaluate_fixed(model, x, y, loss_fn):
    model.eval()
    with torch.no_grad():
        logits = model(x).squeeze(-1)
        loss = float(loss_fn(logits, y))
        score = logits.sigmoid()
        prediction = score >= .5
        correct = int((prediction == y.bool()).sum())
    if not np.isfinite(loss) or not torch.isfinite(score).all():
        raise ValueError('Nonfinite evaluation')
    return {'loss': loss, 'correct': correct, 'n': len(y), 'accuracy': correct/len(y),
            'scores': score.tolist()}


def train_step(model, optimizer, loss_fn, x, y):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(model(x).squeeze(-1), y)
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite training loss')
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    if any(g is None or not torch.isfinite(g).all() for g in grads):
        raise ValueError('Missing/nonfinite gradient')
    grad_norm = sum(float(g.double().square().sum()) for g in grads)**.5
    if grad_norm == 0:
        raise ValueError('Zero gradient')
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Nonfinite model parameters')
    return float(loss.detach()), grad_norm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--preprocess-config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-overfit-steps', type=int, default=500)
    args = parser.parse_args()
    if args.max_overfit_steps < 1:
        raise ValueError('max-overfit-steps must be positive')
    if (args.output_dir/'sensefi_style_check.json').exists():
        raise FileExistsError('Choose a fresh output directory; diagnostic evidence is preserved')
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    seed_all(args.seed)
    dataset = FallDataset(args.manifest, args.preprocess_config, 'train_id')
    loss_fn = nn.BCEWithLogitsLoss()  # each diagnostic subset is balanced: pos_weight=1
    report = {'purpose': 'diagnostics_only_not_generalization', 'seed': args.seed,
              'device': 'cpu', 'torch_version': torch.__version__, 'threads': 2,
              'optimizer': 'Adam', 'learning_rate': .001, 'pos_weight': 1.,
              'threshold': .5, 'official_split': 'train_id',
              'preprocess_config': str(args.preprocess_config), 'validation_or_test_used': False}
    model = SenseFiStyleCNN()
    trace = []
    handles = []
    for name, layer in model.named_modules():
        if name and not list(layer.children()):
            handles.append(layer.register_forward_hook(
                lambda layer, inputs, output, name=name: trace.append(
                    {'name': name, 'layer': repr(layer), 'input': list(inputs[0].shape),
                     'output': list(output.shape)})))
    shape_indices = pick_subset(dataset, 1, args.seed)
    shape_batch = collate_device_segments([dataset[i] for i in shape_indices])
    model.eval()
    with torch.no_grad():
        output = model(shape_batch['x'])
        assert output.shape == (4, 1)
        for device, x in shape_batch['x'].items():
            single = model({device: x})
            offset = 0 if device == 'HP' else len(shape_batch['x'].get('HP', []))
            torch.testing.assert_close(single, output[offset:offset+len(x)], atol=1e-6, rtol=1e-5)
    for handle in handles:
        handle.remove()
    report['layer_trace'] = trace
    report['shape_check'] = {'inputs': {d: list(x.shape) for d,x in shape_batch['x'].items()},
                             'output': list(output.shape), 'labels': shape_batch['y'].tolist(),
                             'device_order': shape_batch['device'], 'passed': True}
    report['trainable_parameters'] = sum(p.numel() for p in model.parameters())
    report['model'] = 'independent SenseFi-style temporal CNN; no pretrained weights'
    print('SHAPES', json.dumps(report['shape_check']), flush=True)
    print('PARAMETERS', report['trainable_parameters'], flush=True)
    # Smoke: 64 complete segments, 16 for each device/label pair, batch 32, 2 epochs.
    smoke_indices = pick_subset(dataset, 16, args.seed)
    subset = Subset(dataset, smoke_indices)
    fixed = next(iter(DataLoader(subset, batch_size=len(subset), collate_fn=collate_device_segments)))
    initial = evaluate_fixed(model, fixed['x'], fixed['y'], loss_fn)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    initial_weight = model.adapters['ESP32'][0].weight.detach().clone()
    loader = DataLoader(subset, batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed), num_workers=0, collate_fn=collate_device_segments)
    smoke_history = []
    for epoch in range(1, 3):
        total, gradients = 0., []
        for batch in loader:
            loss, grad = train_step(model, optimizer, loss_fn, batch['x'], batch['y'])
            total += loss*len(batch['y']); gradients.append(grad)
        metrics = evaluate_fixed(model, fixed['x'], fixed['y'], loss_fn)
        row = {'epoch': epoch, 'train_loss': total/len(subset), 'eval_loss_same_subset': metrics['loss'],
               'accuracy_same_subset': metrics['accuracy'], 'gradient_norms': gradients}
        smoke_history.append(row)
        print('SMOKE', json.dumps(row), flush=True)
    delta = float((model.adapters['ESP32'][0].weight.detach()-initial_weight).abs().max())
    assert delta > 0
    report['smoke'] = {'samples': describe_subset(dataset, smoke_indices), 'n': len(subset),
                       'batch_size': 32, 'epochs': 2, 'optimizer_steps': 4, 'dropout': .3,
                       'initial': initial, 'final': metrics, 'history': smoke_history,
                       'esp32_adapter_max_abs_weight_change': delta, 'passed': True}
    (args.output_dir/'sensefi_style_check.json').write_text(json.dumps(report, indent=2))
    # New model and optimizer: do not initialize from smoke-trained weights.
    seed_all(args.seed)
    model = SenseFiStyleCNN(dropout=0.)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    overfit_indices = pick_subset(dataset, 4, args.seed)
    fixed = next(iter(DataLoader(Subset(dataset, overfit_indices), batch_size=16, collate_fn=collate_device_segments)))
    initial = evaluate_fixed(model, fixed['x'], fixed['y'], loss_fn)
    history = [{'step': 0, **{k: initial[k] for k in ('loss', 'accuracy', 'correct', 'n')}}]
    consecutive = 0
    passed = False
    for step in range(1, args.max_overfit_steps+1):
        training_loss, grad = train_step(model, optimizer, loss_fn, fixed['x'], fixed['y'])
        metrics = evaluate_fixed(model, fixed['x'], fixed['y'], loss_fn)
        history.append({'step': step, 'train_loss': training_loss, 'gradient_norm': grad,
                        **{k: metrics[k] for k in ('loss', 'accuracy', 'correct', 'n')}})
        consecutive = consecutive+1 if metrics['accuracy'] == 1. and metrics['loss'] < .02 else 0
        if step == 1 or step % 25 == 0 or consecutive == 5:
            print('OVERFIT', json.dumps(history[-1]), flush=True)
        if consecutive == 5:
            passed = True
            break
    report['overfit'] = {'samples': describe_subset(dataset, overfit_indices), 'n': len(overfit_indices),
                         'batch_size': 16, 'dropout': 0., 'initial': initial, 'final': metrics,
                         'steps': step, 'max_steps': args.max_overfit_steps, 'passed': passed,
                         'criterion': 'accuracy=1 and BCE<0.02 for 5 consecutive post-update eval checks',
                         'history': history, 'labels': fixed['y'].tolist()}
    report['overfit']['per_device'] = {}
    for device in ('HP', 'ESP32'):
        positions = [i for i,d in enumerate(fixed['device']) if d == device]
        correct = sum((metrics['scores'][i] >= .5) == bool(fixed['y'][i]) for i in positions)
        report['overfit']['per_device'][device] = {'n': len(positions), 'correct': correct}
    report['elapsed_seconds'] = time.monotonic()-started
    (args.output_dir/'sensefi_style_check.json').write_text(json.dumps(report, indent=2))
    with (args.output_dir/'overfit_history.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['step','train_loss','gradient_norm','loss','accuracy','correct','n'])
        writer.writeheader(); writer.writerows(history)
    fig, axes = plt.subplots(1,2,figsize=(10,4),layout='constrained')
    axes[0].plot([h['step'] for h in history],[h['loss'] for h in history])
    axes[0].set_ylabel('BCE on the same 16 training segments')
    axes[1].plot([h['step'] for h in history],[h['accuracy'] for h in history])
    axes[1].set_ylabel('Accuracy on the same 16 training segments'); axes[1].set_ylim(0,1.05)
    for ax in axes:
        ax.set_xlabel('Optimizer update'); ax.grid(alpha=.2)
    fig.suptitle('Small-subset overfit diagnostic — not validation/test performance')
    fig.savefig(args.output_dir/'overfit_curve.png',dpi=150); plt.close(fig)
    print('DONE',json.dumps({'smoke_passed': True,'overfit_passed':passed,'steps':step,
                              'final_loss':metrics['loss'],'correct':metrics['correct'],
                              'elapsed_seconds':report['elapsed_seconds']}),flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
