"""Train the independent subject01 TCS-Fall GCV/CNN pilot on whole-record splits."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import CNN
from prepare import MODULE, RECIPE, sha


def seed_all():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def metrics(labels, scores):
    labels = np.asarray(labels); positive = np.asarray(scores) >= 0.5
    tp = int(((labels == 1) & positive).sum()); fp = int(((labels == 0) & positive).sum())
    fn = int(((labels == 1) & ~positive).sum()); tn = int(((labels == 0) & ~positive).sum())
    return {'precision': tp / (tp + fp) if tp + fp else 0.,
            'recall': tp / (tp + fn) if tp + fn else 0.,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            'accuracy': (tp + tn) / len(labels), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def check_protected():
    old = json.loads((MODULE / 'final_selection_v1/preserved_artifacts_before.json').read_text())
    if any(sha(MODULE / name) != digest for name, digest in old.items()):
        raise ValueError('Earlier protected artifact changed')
    frozen = MODULE / 'outputs/final_selection_v1/final_research_model.pt'
    receipt = json.loads(frozen.with_suffix('.json').read_text())
    if sha(frozen) != receipt['checkpoint_sha256']:
        raise ValueError('Earlier frozen final model changed')
    return {'old_artifacts_checked': len(old), 'old_artifacts_changed': [],
            'old_frozen_final_checkpoint_sha256': receipt['checkpoint_sha256']}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory; preserve previous runs')
    args.output.mkdir(parents=True); args.report.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not torch.backends.mps.is_available():
        raise RuntimeError('Planned MPS run unavailable; no silent backend change')
    device = torch.device('mps')
    prepared = json.loads((args.data_dir / 'preparation.json').read_text())
    split = json.loads((args.data_dir / 'split.json').read_text())
    if prepared['features_sha256'] != sha(args.data_dir / 'features.npz'):
        raise ValueError('Feature cache changed')
    if prepared['split_sha256'] != sha(args.data_dir / 'split.json'):
        raise ValueError('Recording split changed')
    if prepared['native_decoder_sha256'] != sha(MODULE / 'tcs_fall_audit_v1/audit_local_archives.py'):
        raise ValueError('Native decoder changed after feature preparation')
    data = np.load(args.data_dir / 'features.npz', allow_pickle=False)
    x, y, mask, ids = data['x'], data['y'], data['is_validation'], data['ids']
    if x.shape != (207, 1, 400, 90) or not np.isfinite(x).all() or set(y.tolist()) != {0, 1}:
        raise ValueError('Invalid prepared data')
    if len(set(ids[~mask]) & set(ids[mask])):
        raise ValueError('Recording overlap')
    if sorted(ids[mask].tolist()) != split['validation_ids']:
        raise ValueError('Split IDs differ from frozen protocol')
    cfg = {
        'model_version': 'tcs_subject01_gcv_cnn_pilot_v1', 'seed': 42,
        'model': 'unchanged official TCS-Fall CNN class, random initialization',
        'model_config': {'input_shape': [1, 400, 90], 'num_classes': 2, 'dropout': 0.6},
        'optimizer': 'Adam', 'learning_rate': 1e-4, 'weight_decay': 0,
        'batch_size': 8, 'batch_size_reason': 'local Apple M2 8GiB; smaller than official32 for activation memory',
        'max_epochs': 100, 'early_stop_patience': 10, 'checkpoint_selection': 'strictly lowest validation loss',
        'criterion': 'CrossEntropyLoss, unweighted as public TCS training code',
        'class_weight_reason': 'training72 Fall/93 Nonfall; retain unweighted public-code loss for pilot',
        'threshold': 0.5, 'label_mapping': {'Nonfall': 0, 'Fall': 1},
        'preprocess_config': RECIPE, 'normalization_parameters': None,
        'features_sha256': prepared['features_sha256'], 'split_sha256': prepared['split_sha256'],
        'data_version': {'archive_sha256': prepared['archive_sha256'], 'scope': 'single nominal subject01,207 intact SA records'},
        'split_identifiers': split, 'backend': 'mps', 'dtype': 'float32',
        'train_fall_nonfall': prepared['train_fall_nonfall'],
        'validation_fall_nonfall': prepared['validation_fall_nonfall'],
        'versions': {'torch': str(torch.__version__), 'numpy': str(np.__version__)},
        'source_sha256': {name: sha(Path(__file__).parent / name) for name in ('model.py', 'prepare.py', 'train.py')},
        'native_decoder_sha256': prepared['native_decoder_sha256'],
        'protected_before': check_protected(), 'continuous_evaluated': False, 'test_evaluated': False,
        'scientific_scope': 'within-subject feasibility experiment; no cross-subject/session generalization claim',
    }
    write_json(args.report / 'run_config.json', cfg)
    write_json(args.output / 'run_config.json', cfg)

    seed_all()
    model = CNN().to(device)
    layer_shapes = {}
    hooks = [getattr(model, name).register_forward_hook(
        lambda layer, inputs, output, name=name: layer_shapes.update({name: list(output.shape)}))
        for name in ('conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'linear')]
    model.eval()
    with torch.inference_mode():
        output = model(torch.from_numpy(x[:1]).to(device))
    for hook in hooks:
        hook.remove()
    if output.shape != (1, 2) or not torch.isfinite(output).all():
        raise ValueError('Shape/finite forward check failed')
    train_indices = np.flatnonzero(~mask)
    smoke_indices = np.concatenate([train_indices[y[train_indices] == label][:4] for label in (0, 1)])
    model.train(); smoke_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    smoke_losses = []
    for _ in range(2):
        smoke_optimizer.zero_grad(set_to_none=True)
        logits = model(torch.from_numpy(x[smoke_indices]).to(device))
        loss = nn.functional.cross_entropy(logits, torch.from_numpy(y[smoke_indices]).to(device))
        loss.backward()
        if not torch.isfinite(loss) or any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise ValueError('Smoke backward finite check failed')
        smoke_optimizer.step(); smoke_losses.append(float(loss.item()))
    shape_report = {'input': [1, 1, 400, 90], 'layer_outputs': layer_shapes,
                    'parameter_count': sum(p.numel() for p in model.parameters()),
                    'smoke_training_steps': 2, 'smoke_batch_shape': [8, 1, 400, 90],
                    'smoke_labels': y[smoke_indices].tolist(), 'smoke_losses': smoke_losses,
                    'smoke_gradients_finite': True,
                    'formal_training_reinitialized_after_smoke': True}
    write_json(args.report / 'shape_smoke_check.json', shape_report)
    print(json.dumps(shape_report), flush=True)
    del model, smoke_optimizer, loss, logits
    torch.mps.empty_cache()
    seed_all()  # Smoke updates never become formal initialization.
    model = CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    train = TensorDataset(torch.from_numpy(x[~mask]), torch.from_numpy(y[~mask]))
    validation = TensorDataset(torch.from_numpy(x[mask]), torch.from_numpy(y[mask]))
    generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train, batch_size=8, shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(validation, batch_size=8, shuffle=False, num_workers=0)
    best_loss, best_epoch, stale = float('inf'), 0, 0
    history = []
    best_scores = None
    for epoch in range(1, 101):
        epoch_start = time.perf_counter()
        model.train(); loss_sum = 0.; train_labels = []; train_scores = []
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(bx); loss = criterion(logits, by)
            if not torch.isfinite(loss).item():
                raise ValueError('Nonfinite training loss; no automatic repair/restart')
            loss.backward(); optimizer.step()
            loss_sum += float(loss.item()) * len(by)
            train_labels.extend(by.detach().cpu().tolist())
            train_scores.extend(logits.detach().softmax(1)[:, 1].cpu().tolist())
        model.eval(); val_loss_sum = 0.; labels = []; scores = []
        with torch.inference_mode():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx); val_loss = criterion(logits, by)
                if not torch.isfinite(logits).all() or not torch.isfinite(val_loss):
                    raise ValueError('Nonfinite validation output')
                val_loss_sum += float(val_loss.item()) * len(by)
                labels.extend(by.cpu().tolist()); scores.extend(logits.softmax(1)[:, 1].cpu().tolist())
        train_metrics, val_metrics = metrics(train_labels, train_scores), metrics(labels, scores)
        val_mean = val_loss_sum / len(validation)
        record = {'epoch': epoch, 'train_loss': loss_sum / len(train), 'val_loss': val_mean,
                  **{f'val_{k}': v for k, v in val_metrics.items()}, 'train_accuracy': train_metrics['accuracy'],
                  'epoch_seconds': time.perf_counter() - epoch_start}
        history.append(record)
        if val_mean < best_loss:
            best_loss, best_epoch, stale = val_mean, epoch, 0
            best_scores = scores.copy()
            checkpoint = {'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                          'model_config': cfg['model_config'], 'preprocess_config': RECIPE,
                          'normalization_parameters': None, 'label_mapping': cfg['label_mapping'],
                          'seed': 42, 'best_epoch': epoch, 'validation_loss': best_loss,
                          'validation_metrics_at_0_5': val_metrics, 'threshold': 0.5,
                          'model_version': cfg['model_version'], 'data_version': cfg['data_version'],
                          'split_identifiers': split, 'run_config': cfg}
            temporary = args.output / 'best.pt.tmp'
            torch.save(checkpoint, temporary); temporary.replace(args.output / 'best.pt')
            with (args.report / 'best_validation_predictions.csv').open('w', newline='') as stream:
                writer = csv.writer(stream); writer.writerow(['sample_id', 'label', 'fall_score', 'prediction_at_0_5'])
                writer.writerows((i, int(yv), float(s), int(s >= 0.5)) for i, yv, s in zip(ids[mask], labels, scores))
        else:
            stale += 1
        with (args.report / 'history.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(record)); writer.writeheader(); writer.writerows(history)
        write_json(args.report / 'progress.json', {'epoch': epoch, 'best_epoch': best_epoch,
                   'best_val_loss': best_loss, 'epochs_no_improvement': stale, 'latest': record})
        print(json.dumps({**record, 'best_epoch': best_epoch, 'stale': stale}), flush=True)
        if stale >= 10:
            break
    checkpoint = torch.load(args.output / 'best.pt', map_location='cpu', weights_only=True)
    summary = {
        'model_version': cfg['model_version'], 'seed': 42, 'epochs_run': len(history),
        'best_epoch': best_epoch, 'best_val_loss': best_loss,
        'best_validation_metrics_at_0_5': checkpoint['validation_metrics_at_0_5'],
        'train_samples': len(train), 'validation_samples': len(validation),
        'train_fall_nonfall': prepared['train_fall_nonfall'], 'validation_fall_nonfall': prepared['validation_fall_nonfall'],
        'elapsed_seconds': time.perf_counter() - start, 'stopped_by_patience': stale >= 10,
        'checkpoint': str((args.output / 'best.pt').resolve()), 'checkpoint_sha256': sha(args.output / 'best.pt'),
        'parameter_count': shape_report['parameter_count'], 'protected_after': check_protected(),
        'test_evaluated': False, 'continuous_evaluated': False,
        'completed_utc': datetime.now(timezone.utc).isoformat(),
    }
    write_json(args.report / 'training_summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
