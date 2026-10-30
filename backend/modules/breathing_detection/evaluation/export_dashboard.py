"""Export a compact, real-data offline dashboard without changing model weights."""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

from backend.shared.export import write_replay
from backend.modules.breathing_detection.preprocessing.wire_breath import WireBreathDataset
from backend.modules.breathing_detection.preprocessing.wire_cache import load_wire_breath_cache
from backend.modules.breathing_detection.training.lstm import LightweightBreathingLSTM
from .evaluate_wire_breath import predict
from .export_unified_rate_dashboard import apply_unified_rate

ROOT = Path('backend/modules/breathing_detection/outputs')
SCENES = [
    ('Quiet reference', 'Subject03 · Foetus · Quiet', 'Room2/Subject03/Foetus/Quiet/0817/Trial01'),
    ('Motion interference', 'Subject02 · Yearner · Interference', 'Room2/Subject02/Yearner/Interference/0816/Trial01'),
    ('Detection challenge', 'Subject02 · Sitting · Quiet', 'Room2/Subject02/Sitting/Quiet/0816/Trial01'),
    ('Sampling gaps', 'Subject01 · Foetus · Interference', 'Room1/Subject01/Foetus/Interference/no-session/Trial01'),
    ('Competing rate peaks', 'Subject05 · Yearner · Interference', 'Room2/Subject05/Yearner/Interference/0817/Trial01'),
]


def read(relative):
    return json.loads((ROOT / relative).read_text())


def waveform(values):
    values = np.asarray(values, dtype=float)
    values = (values - values.mean()) / max(values.std(), 1e-8)
    return np.round(np.interp(np.linspace(0, len(values) - 1, 180), np.arange(len(values)), values), 4).tolist()


def main():
    cache = load_wire_breath_cache(ROOT / 'wire_breath_cache', 1401)
    checkpoint = torch.load(ROOT / 'lstm_interpolated_full_seed123/best_model.pt', map_location='cpu', weights_only=True)
    model = LightweightBreathingLSTM(input_size=232, hidden_size=64, projection_size=32)
    model.load_state_dict(checkpoint['model_state'])
    probabilities = predict(model, cache.sequences, torch.device('cpu'), 128)
    windows = {row['sample_id']: (row, float(probabilities[i])) for i, row in enumerate(cache.manifest['samples'])}
    rates = {r['sample_id']: r for r in read('respiratory_rate_baseline_v1/window_estimates.json')['30']}
    candidates = {r['sample_id']: r for r in read('respiratory_rate_boundary_only_v1/window_predictions.json')}
    dataset = WireBreathDataset('data/WiReBreath', window_seconds=30, stride_seconds=30, normalize=False)
    records = {scene[2]: dict(id=scene[2], title=scene[0], description=scene[1], samples=[]) for scene in SCENES}
    threshold = float(checkpoint['threshold'])
    for i in range(len(dataset)):
        ref = dataset._windows[i]
        if ref.recording_id not in records:
            continue
        sample = dataset[i]
        raw_time = np.asarray(loadmat(ref.path, squeeze_me=True, variable_names=('time',))['time'], dtype=float).reshape(-1)
        differences = np.diff(raw_time)
        sampling_gaps = [dict(start=round(max(0., float(raw_time[j]) - ref.start_seconds), 6),
                              end=round(min(30., float(raw_time[j + 1]) - ref.start_seconds), 6))
                         for j in np.flatnonzero(differences > .1)
                         if raw_time[j] < ref.start_seconds + 30 and raw_time[j + 1] > ref.start_seconds]
        rate = rates[sample.sample_id]
        prefix = sample.sample_id.split('@')[0]
        evidence = []
        for seconds in (0, 10, 20):
            window, probability = windows[f'{prefix}@{seconds:.3f}s']
            evidence.append(dict(start=seconds, probability=round(probability, 6), assessable=window['assessable']))
        last = evidence[-1]
        status = 'not_assessable' if not last['assessable'] else 'breathing' if last['probability'] >= threshold else 'none'
        p = last['probability']
        margin = abs(p - threshold) / ((1 - threshold) if p >= threshold else threshold)
        candidate = candidates.get(sample.sample_id)
        records[ref.recording_id]['samples'].append(dict(
            id=sample.sample_id, segment=sample.metadata['segment_id'], room=sample.metadata['room'],
            subject=sample.metadata['participant_id'], posture=sample.metadata['posture'], environment=sample.metadata['environment'],
            status=status, confidence=round(margin, 5) if last['assessable'] else 0,
            evidence=evidence, quality_flags=list(sample.quality_flags), rate_assessable=sample.assessable,
            sampling_gaps=sampling_gaps,
            baseline_bpm=rate['estimates']['mean_spectrum'] if rate['estimates'] else None,
            candidate_bpm=candidate['predictions']['selected'] if candidate else None,
            reference_bpm=rate['reference_bpm'],
            csi_waveform=waveform(sample.x[0].mean(axis=0)), reference_waveform=waveform(sample.respiratory_reference),
        ))
    frames = []
    for record in records.values():
        record['samples'].sort(key=lambda s: s['segment'])
        for sample in record['samples']:
            frames.append(dict(t=len(frames) * 30, status=sample['status'], confidence=sample['confidence'], sample_id=sample['id']))
    rate_report = read('respiratory_rate_boundary_only_v1/metrics.json')
    payload = dict(module='breathing_detection', version='0.2', mock=False, mode='offline_research',
                   timeline_semantics='Independent recorded samples; frame t is a browsing coordinate, not synchronized sensor time.',
                   frames=frames, records=list(records.values()), threshold=threshold,
                   provenance=dict(dataset='WiRe-Breath', checkpoint='CSI-Bench seed 123 · epoch 28',
                                   rate_method='30s spectral baseline / fold-selected boundary candidate',
                                   reliability_gate_enabled=False, selection='Five curated diagnostic records; not a representative performance sample.'),
                   benchmarks=dict(csi_bench_test=dict(macro_f1=.9670, recall=.9909, fpr=.0673, tn=2716, fp=196, fn=110, tp=11977),
                                   external_recall=.9721577726, rate=rate_report['selected'],
                                   rate_person_mae=rate_report['person_macro_mae_bpm'], rate_coverage=rate_report['total_window_coverage']),
                   experiments=[dict(name='Frozen LSTM', result='97.22% external recall', decision='Reference', note='Positive-only external evaluation; specificity unavailable.'),
                                dict(name='Frequency augmentation', result='94.51% external recall', decision='Not promoted', note='Synthetic robustness improved, external recall declined.'),
                                dict(name='Boundary-only rate correction', result='0.82 BPM pooled MAE', decision='Experimental', note='394 valid 30s windows; 22 large errors remain.'),
                                dict(name='Rate reliability screen', result='2 / 362 accurate rates rejected', decision='Disabled', note='Subject05 rejection 2.13% exceeds the 1% limit.')])
    write_replay('breathing_detection', apply_unified_rate(payload), bundle=True)
    print('Exported', len(records), 'records and', len(frames), 'real-data samples')


if __name__ == '__main__':
    main()
