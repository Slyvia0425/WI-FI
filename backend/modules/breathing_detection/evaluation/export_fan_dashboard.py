"""Append score-blind real fan examples, preserving existing replay content."""
import json
from pathlib import Path

import numpy as np

from backend.shared.export import write_replay
from ..preprocessing.data import BreathingDataset
from .evaluate_fan_detection import SCENARIOS, CHECKPOINT_SHA, THRESHOLD
from .evaluate_spike_preprocessing import ROOT, digest


def select_fan_ids(rows):
    """One first-ID record per device/scenario/class; never inspect model scores."""
    groups = {}
    for row in sorted(rows, key=lambda r: r['sample_id']):
        key = (row['scenario'], row['label'], row['device_id'])
        if row['probability'] is not None:
            groups.setdefault(key, row['sample_id'])
    return groups


def main():
    folder = ROOT/'fan_detection_v1'
    config = json.loads((folder/'config.json').read_text())
    predictions = json.loads((folder/'predictions.json').read_text())
    metrics = json.loads((folder/'metrics.json').read_text())
    if config['checkpoint_sha256'] != CHECKPOINT_SHA or config['threshold'] != THRESHOLD:
        raise ValueError('Fan assessment checkpoint/threshold mismatch')
    source = BreathingDataset('data/BreathingDetection', 'test_id', normalize=False)
    indices = {r['id']: i for i, r in enumerate(source._rows)}
    by_id = {r['sample_id']: r for r in predictions}
    selected = select_fan_ids(predictions)
    scenes = []
    for scenario in SCENARIOS:
        los, fan = scenario.split('_')
        for label, presence in ((1, 'Breathing present'), (0, 'Empty room')):
            ids = [sample_id for (s, y, _), sample_id in selected.items() if s == scenario and y == label]
            if not ids:
                continue
            samples = []
            for sample_id in ids:
                sample = source[indices[sample_id]]
                row = by_id[sample_id]
                raw_path = str(Path('data/BreathingDetection')/source._rows[indices[sample_id]]['file_path'])
                if not sample.assessable or digest(raw_path) != config['raw_file_sha256'][raw_path]:
                    raise ValueError('Selected fan source changed')
                wave = sample.x[0].mean(axis=0).astype(float)
                wave = (wave-wave.mean())/max(float(wave.std()), 1e-8)
                samples.append({'id': sample_id, 'label': label, 'label_name': row['label_name'],
                    'probability': row['probability'], 'status': 'breathing' if row['prediction'] else 'none',
                    'waveform': np.round(wave, 5).tolist(), 'time_steps': 300, 'nominal_rate_hz': 30,
                    'duration_seconds': 10, 'split': 'test_id', 'metadata': sample.metadata})
            state = 'Fan on' if fan == 'FanOn' else 'Fan off'
            scenes.append({'id': f'{scenario}_{label}', 'title': f'{state} · {presence} · {los}',
                           'scenario': scenario, 'fan_on': fan == 'FanOn', 'los': los == 'LOS',
                           'label': label, 'samples': samples})
    payload = json.loads((ROOT/'replay.json').read_text())
    before = json.loads(json.dumps(payload))
    payload['csi_bench']['fan_experience'] = {'version': 1, 'scenes': scenes, 'threshold': THRESHOLD,
        'checkpoint_sha256': CHECKPOINT_SHA, 'split': 'test_id',
        'selection': 'First assessable sample ID per device/scenario/class; selection before score inspection, not prediction correctness.',
        'assessment': {'sample_count': 4156, 'by_scenario': {s: {k: m[k] for k in (
            'available_classes', 'confusion_matrix', 'breathing_recall', 'empty_false_positive_rate')}
            for s, m in metrics['by_scenario'].items()}},
        'assessment_sha256': digest(folder/'metrics.json'),
        'limitations': 'Observed U23/E09 test subset, not independent external validation. LOS fan-on empty class missing. No respiratory-rate reference. Different records, not paired on/off captures.'}
    # Only the additive fan key may change; prior CSI examples/rates/streams survive.
    check = json.loads(json.dumps(payload))
    check['csi_bench'].pop('fan_experience')
    before['csi_bench'].pop('fan_experience', None)
    if check != before:
        raise ValueError('Existing replay data would be altered')
    write_replay('breathing_detection', payload, bundle=True)
    print('Exported', len(scenes), 'fan scenarios and', sum(len(s['samples']) for s in scenes), 'real samples; prior data preserved.')


if __name__ == '__main__':
    main()
