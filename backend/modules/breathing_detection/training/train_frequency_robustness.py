"""Paired clean/augmented fine-tuning pilot with validation-only decisions."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from backend.modules.breathing_detection.evaluation.evaluate_lstm import checkpoint_sha256
from backend.modules.breathing_detection.evaluation.perturb_wire_frequency import perturb, score
from backend.modules.breathing_detection.preprocessing.cache import load_temporal_cache
from backend.modules.breathing_detection.training.lstm import LightweightBreathingLSTM
from backend.modules.breathing_detection.training.periodicity_baseline import binary_metrics
from backend.modules.breathing_detection.training.train_lstm import set_seed


def augment_batch(values, rng):
    """Apply an independent fixed-range edge shift to each window, both classes."""
    shifts = rng.choice([-4, -1, 0, 1, 4], size=len(values))
    return np.stack([perturb(row, int(shift), edge=True) for row, shift in zip(values, shifts)])


def stratified_indices(cache, count, rng):
    available = np.flatnonzero(np.asarray(cache.weights) > 0)
    labels = np.asarray(cache.labels)
    positives = int(round(count * np.mean(labels[available])))
    return np.concatenate([
        rng.choice(available[labels[available] == label], n, replace=False)
        for label, n in ((0, count - positives), (1, positives))
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--train-count', type=int, default=6000)
    parser.add_argument('--epochs', type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / 'config.json').exists():
        raise FileExistsError('Use a new output directory to preserve this experiment')
    config = {**vars(args), 'seed': 20261028, 'learning_rate': 1e-4,
              'threshold_policy': 'original frozen threshold',
              'augmentation': 'independent edge shifts sampled uniformly from [-4,-1,0,1,4]',
              'promotion_limits': {'macro_f1_drop_max': .005, 'FPR_increase_max': .005, 'recall_drop_max': .005},
              'robustness_requirement': 'mean shifted-validation macro-F1 improves against BOTH frozen and clean-finetuned control'}
    (args.output_dir / 'config.json').write_text(json.dumps(config, indent=2, default=str))
    torch.set_num_threads(4)
    base = Path('backend/modules/breathing_detection/outputs')
    checkpoint_path = base / 'lstm_interpolated_full_seed123/best_model.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    train = load_temporal_cache(base / 'temporal_cache', 'train_id', 70000)
    val = load_temporal_cache(base / 'temporal_cache', 'val_id', 15000)
    rng = np.random.default_rng(20261028)
    indices = stratified_indices(train, args.train_count, rng)
    validation_indices = np.flatnonzero(np.asarray(val.weights) > 0)
    stress_indices = stratified_indices(val, 2000, rng)
    np.savez(args.output_dir / 'sample_indices.npz', train=indices, validation_stress=stress_indices)
    x = np.asarray(train.sequences[indices], dtype=np.float32)
    y = np.asarray(train.labels[indices], dtype=np.float32)
    validation_x = val.sequences
    validation_y = np.asarray(val.labels[validation_indices])
    stress_x = np.asarray(val.sequences[stress_indices], dtype=np.float32)
    stress_y = np.asarray(val.labels[stress_indices])
    threshold = float(checkpoint['threshold'])
    results = {}
    for arm in ('frozen', 'clean_finetune', 'augmented_finetune'):
        set_seed(20261028)
        order_rng = np.random.default_rng(20261028)
        augmentation_rng = np.random.default_rng(20261029)
        model = LightweightBreathingLSTM(input_size=232, hidden_size=64, projection_size=32)
        model.load_state_dict(copy.deepcopy(checkpoint['model_state']))
        losses = []
        if arm != 'frozen':
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(np.sum(y == 0) / np.sum(y == 1))))
            for epoch in range(args.epochs):
                model.train()
                order = order_rng.permutation(len(x))
                total = 0.
                for start in range(0, len(x), 128):
                    batch_indices = order[start:start + 128]
                    batch = x[batch_indices]
                    if arm == 'augmented_finetune':
                        batch = augment_batch(batch, augmentation_rng)
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(torch.from_numpy(batch)), torch.from_numpy(y[batch_indices]))
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    total += float(loss.detach()) * len(batch)
                losses.append(total / len(x))
                print(arm, 'epoch', epoch + 1, 'loss', losses[-1], flush=True)
        model.eval()
        clean = binary_metrics(validation_y, score(model, validation_x)[validation_indices], threshold)
        stress = {}
        for shift in (-4, -1, 1, 4):
            stress[str(shift)] = binary_metrics(stress_y, score(model, stress_x, shift=shift, edge=True), threshold)
        results[arm] = {'clean_validation': clean, 'shifted_validation': stress,
                        'mean_shifted_macro_f1': float(np.mean([v['macro_f1'] for v in stress.values()])),
                        'training_losses': losses}
        if arm != 'frozen':
            candidate = {**checkpoint, 'model_state': model.state_dict(), 'epoch': args.epochs,
                         'parent_checkpoint_sha256': checkpoint_sha256(checkpoint_path), 'experiment': arm}
            torch.save(candidate, args.output_dir / f'{arm}.pt')
        print(arm, json.dumps(results[arm]), flush=True)
    frozen = results['frozen']['clean_validation']
    augmented = results['augmented_finetune']['clean_validation']
    checks = {
        'clean_macro_f1': augmented['macro_f1'] >= frozen['macro_f1'] - .005,
        'clean_FPR': augmented['empty_room_false_positive_rate'] <= frozen['empty_room_false_positive_rate'] + .005,
        'clean_recall': augmented['recall'] >= frozen['recall'] - .005,
        'robustness_vs_both_controls': results['augmented_finetune']['mean_shifted_macro_f1'] > max(
            results['frozen']['mean_shifted_macro_f1'], results['clean_finetune']['mean_shifted_macro_f1']),
    }
    report = {'parent_sha256': checkpoint_sha256(checkpoint_path), 'threshold': threshold,
              'train_class_counts': {'empty': int(np.sum(y == 0)), 'breathing': int(np.sum(y == 1))},
              'validation_count': len(validation_y), 'results': results,
              'promotion_checks': checks, 'passes_pilot': all(checks.values()),
              'test_accessed': False, 'wire_breath_accessed': False}
    (args.output_dir / 'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'promotion_checks': checks, 'passes_pilot': all(checks.values())}), flush=True)


if __name__ == '__main__':
    main()
