"""Official train/validation CNN training. Never constructs a test dataset."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

if __package__:
    from .dataset import FallDataset
    from .models.cnn import FallCNN
else:
    from dataset import FallDataset
    from models.cnn import FallCNN


def binary_metrics(labels, predictions):
    y = np.asarray(labels, dtype=bool)
    p = np.asarray(predictions, dtype=bool)
    tp, fp = int((y & p).sum()), int((~y & p).sum())
    fn, tn = int((y & ~p).sum()), int((~y & ~p).sum())
    precision = tp/(tp+fp) if tp+fp else 0.
    recall = tp/(tp+fn) if tp+fn else 0.
    f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    return dict(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn)


def validate(model, loader, criterion, device):
    model.eval()
    total, count = 0., 0
    labels, predictions, records = [], [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch['x'].to(device), batch['y'].to(device)
            logits = model(x).squeeze(-1)
            loss = criterion(logits, y)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise ValueError('Nonfinite validation output/loss')
            scores = logits.sigmoid().tolist()
            labels.extend(batch['y'].tolist())
            predictions.extend((logits >= 0).tolist())
            total += float(loss)*len(logits); count += len(logits)
            records.extend(dict(sample_id=sid, device=d, label=int(y), fall_score=s)
                           for sid, d, y, s in zip(batch['sample_id'], batch['device'], batch['y'].tolist(), scores))
    if count != len(loader.dataset):
        raise ValueError('Incomplete validation coverage')
    return total/count, binary_metrics(labels, predictions), records


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--preprocess-config', type=Path, required=True)
    parser.add_argument('--audit-details', type=Path, required=True)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--device', choices=['cpu','mps'], default='cpu')
    args = parser.parse_args()
    if min(args.max_epochs, args.patience, args.batch_size) < 1 or args.learning_rate <= 0:
        raise ValueError('Training settings must be positive')
    for folder in (args.checkpoint_dir, args.report_dir):
        folder.mkdir(parents=True, exist_ok=True)
    best_path = args.checkpoint_dir/'best.pt'
    if best_path.exists() or (args.report_dir/'history.csv').exists():
        raise FileExistsError('Run already exists; choose fresh directories to preserve results')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS unavailable; specify --device cpu')
    if args.device == 'mps':
        torch.mps.manual_seed(args.seed)
    train = FallDataset(args.manifest, args.preprocess_config, 'train_id')
    val = FallDataset(args.manifest, args.preprocess_config, 'val_id')
    train_ids = {r['sample_id'] for r in train.rows}
    val_ids = {r['sample_id'] for r in val.rows}
    if train_ids & val_ids:
        raise ValueError('Train/validation ID overlap')
    if train_ids != set(train.preprocessor.config['training_sample_ids']):
        raise ValueError('Normalization was not fitted on this training membership')
    positives = sum(int(r['binary_label']) for r in train.rows)
    negatives = len(train)-positives
    if not positives or not negatives:
        raise ValueError('Both training classes required')
    pos_weight = negatives/positives
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=generator,
                              num_workers=0, drop_last=False)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    model = FallCNN(in_features=train.preprocessor.config['target_features'], dropout=.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    audit = json.loads(args.audit_details.read_text())
    config = dict(seed=args.seed, max_epochs=args.max_epochs, patience=args.patience, min_delta=0.,
                  batch_size=args.batch_size, learning_rate=args.learning_rate, optimizer='Adam',
                  dropout=.3, device=args.device, threads=2, deterministic_algorithms=True,
                  train_n=len(train), val_n=len(val), train_fall=positives, train_nonfall=negatives,
                  val_fall=sum(int(r['binary_label']) for r in val.rows),
                  pos_weight=pos_weight, pos_weight_formula='train_nonfall/train_fall',
                  validation_loss='same train-derived weighted BCE as training',
                  checkpoint_selection='strictly lowest validation loss', fixed_metric_threshold=.5,
                  test_evaluated=False, threshold_optimized=False,
                  data_version=audit['data_version'], torch_version=str(torch.__version__),
                  manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest())
    atomic_json(args.report_dir/'run_config.json', config)
    print('CONFIG', json.dumps(config), flush=True)
    history = []
    best_loss, best_epoch, wait = float('inf'), 0, 0
    started = time.monotonic()
    for epoch in range(1, args.max_epochs+1):
        epoch_started = time.monotonic()
        model.train()
        total, count = 0., 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            x, y = batch['x'].to(device), batch['y'].to(device)
            logits = model(x).squeeze(-1)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Missing/nonfinite gradient')
            optimizer.step()
            total += float(loss.detach())*len(logits); count += len(logits)
        if count != len(train):
            raise ValueError('Incomplete training coverage')
        val_loss, metrics, records = validate(model, val_loader, criterion, device)
        improved = val_loss < best_loss
        if improved:
            best_loss, best_epoch, wait = val_loss, epoch, 0
            preprocess_config = train.preprocessor.config
            checkpoint = {
                'model_version': 'cnn_baseline_v1', 'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'model_config': model.config, 'preprocess_config': preprocess_config,
                'normalization_parameters': preprocess_config['devices'],
                'label_mapping': {'Fall': 1, 'Nonfall': 0},
                'official_label_mapping': audit['official_label_mapping'],
                'seed': args.seed, 'best_epoch': epoch, 'best_val_loss': val_loss,
                'validation_metrics_at_0_5': metrics, 'training_config': config,
                'data_version': audit['data_version'], 'data_content_sha256': audit['data_content_sha256'],
                'manifest_sha256': config['manifest_sha256'],
                'split_identifiers': {'train_id': sorted(train_ids), 'val_id': sorted(val_ids)},
                'metadata_split_sha256': audit['metadata_split_sha256'],
                'optimizer_state_dict': optimizer.state_dict(),
                'metric_threshold': .5, 'selected_threshold': None,
                'threshold_selection_status': 'not_performed',
                'known_limitations': ['official train/validation content duplicates',
                                      'device confounding and positional zero padding',
                                      'physical link semantics and sampling rate unverified'],
            }
            tmp = best_path.with_suffix('.pt.tmp')
            torch.save(checkpoint, tmp); os.replace(tmp, best_path)
            with (args.report_dir/'best_validation_predictions.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['sample_id','device','label','fall_score'])
                writer.writeheader(); writer.writerows(records)
        else:
            wait += 1
        row = dict(epoch=epoch, train_loss=total/count, val_loss=val_loss,
                   **metrics, improved=improved, wait=wait, seconds=time.monotonic()-epoch_started)
        history.append(row)
        with (args.report_dir/'history.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader(); writer.writerows(history)
        atomic_json(args.report_dir/'progress.json', dict(last_epoch=epoch, best_epoch=best_epoch,
                    best_val_loss=best_loss, wait=wait, completed=False))
        print('EPOCH', json.dumps(row), flush=True)
        if wait >= args.patience:
            break
    best_row = next(row for row in history if row['epoch'] == best_epoch)
    summary = dict(config=config, completed=True, epochs_run=len(history), best_epoch=best_epoch,
                   best_val_loss=best_loss, best_epoch_results=best_row, last_epoch_results=history[-1],
                   stop_reason='early_stopping' if wait >= args.patience else 'max_epochs',
                   elapsed_seconds=time.monotonic()-started, checkpoint=str(best_path),
                   test_evaluated=False, threshold_optimized=False)
    atomic_json(args.report_dir/'training_summary.json', summary)
    atomic_json(args.report_dir/'progress.json', summary)
    fig, axes = plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    epochs = [h['epoch'] for h in history]
    axes[0].plot(epochs,[h['train_loss'] for h in history],label='Training (Dropout on)')
    axes[0].plot(epochs,[h['val_loss'] for h in history],label='Validation (Dropout off)')
    axes[0].set_ylabel('Train-weighted BCE'); axes[0].legend()
    for key in ('precision','recall','f1'):
        axes[1].plot(epochs,[h[key] for h in history],label=key.capitalize())
    axes[1].set_ylabel('Validation metric at fixed threshold 0.5');axes[1].set_ylim(0,1.05);axes[1].legend()
    for ax in axes:
        ax.axvline(best_epoch,color='gray',ls='--',label='Best validation loss')
        ax.set_xlabel('Epoch');ax.grid(alpha=.2)
    fig.suptitle('Official train/validation split — known content overlap; no test evaluation')
    fig.savefig(args.report_dir/'training_curves.png',dpi=150);plt.close(fig)
    print('DONE',json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
