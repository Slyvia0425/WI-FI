"""Export deterministic official-test examples, without fitting or score selection."""
import json
from pathlib import Path
import numpy as np
import torch
from backend.shared.export import write_replay
from ..inference.frozen_detector import FrozenBreathingDetector
from ..preprocessing.data import BreathingDataset
from ..preprocessing.temporal import extract_interpolated_subcarriers
from .evaluate_lstm import checkpoint_sha256

ROOT = Path('backend/modules/breathing_detection/outputs')


def select_examples(source, per_class=3):
    """Choose assessable samples by ID, never by model score or correctness."""
    selected = {0: [], 1: []}
    for index in sorted(range(len(source)), key=lambda i: source._rows[i]['id']):
        label = {'empty': 0, 'sleep': 1}[source._rows[index]['label']]
        if len(selected[label]) >= per_class:
            continue
        sample = source[index]
        if sample.assessable:
            selected[label].append(sample)
        if all(len(group) == per_class for group in selected.values()):
            break
    if any(len(group) < per_class for group in selected.values()):
        raise ValueError('Not enough assessable examples in each class')
    return selected


def main():
    torch.set_num_threads(4)
    payload = json.loads((ROOT / 'replay.json').read_text())
    checkpoint = ROOT / 'lstm_interpolated_full_seed123/best_model.pt'
    detector = FrozenBreathingDetector(checkpoint)
    source = BreathingDataset('data/BreathingDetection', 'test_id', normalize=False)
    selected = select_examples(source)
    scenes = []
    for label, title in [(1, 'Breathing present'), (0, 'Empty room')]:
        samples = []
        for sample in selected[label]:
            p = detector(extract_interpolated_subcarriers(sample.x))
            mean = sample.x[0].mean(axis=0).astype(float)
            wave = (mean - mean.mean()) / max(float(mean.std()), 1e-8)
            samples.append(dict(id=sample.sample_id, label=label, label_name=sample.metadata['label_name'],
                probability=p, status='breathing' if p >= detector.threshold else 'none',
                waveform=np.round(wave, 5).tolist(), time_steps=300, nominal_rate_hz=30,
                duration_seconds=10, split=sample.split, metadata=sample.metadata))
        scenes.append(dict(id='breathing' if label else 'empty', title=title, samples=samples))
    payload['csi_bench'] = dict(version=1, scenes=scenes, threshold=detector.threshold,
        checkpoint_sha256=checkpoint_sha256(checkpoint), split='test_id',
        selection='First three assessable IDs per class in lexicographic order; selection before inference, not score-filtered.',
        timing='300 stored samples at nominal 30 Hz; original packet timestamps unavailable. Independent 10s samples, not continuous streaming.',
        limitations='Previously observed official test; illustrative examples, not a new holdout or performance estimate. No reference respiration or 30s rate estimate. No threshold/model changes.')
    write_replay('breathing_detection', payload, bundle=True)
    print(json.dumps({scene['title']: [(s['id'], s['status'], round(s['probability'], 6)) for s in scene['samples']] for scene in scenes}, indent=2))


if __name__ == '__main__':
    main()
