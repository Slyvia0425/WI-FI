"""Large-rate-error diagnostics and inner-validation-only confidence screening."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.signal import detrend, find_peaks

from backend.modules.breathing_detection.preprocessing.respiratory_rate import estimate_rates, spectra
from backend.modules.breathing_detection.preprocessing.wire_breath import WireBreathDataset
from backend.modules.breathing_detection.preprocessing.wire_cache import load_wire_breath_cache
from backend.modules.breathing_detection.preprocessing.wire_protocol import DEFAULT_PATH, load_protocol

FEATURES = ('motion_ratio', 'negative_peak_concentration', 'negative_peak_ratio', 'negative_agreement', 'temporal_spread')


def features(x):
    frequencies, power = spectra(x[0])
    mean = power.mean(axis=0)
    main = int(np.argmax(mean))
    main_hz = frequencies[main]
    concentration = float(mean[np.abs(frequencies - main_hz) <= 1 / 30].sum() / mean.sum())
    peaks, _ = find_peaks(mean)
    other = [mean[i] for i in peaks if abs(frequencies[i] - main_hz) > 2 / 30]
    ratio = float(mean[main] / max(max(other, default=0), 1e-12))
    agreement = float(np.mean(abs(frequencies[np.argmax(power, axis=1)] - main_hz) <= 1 / 30))
    values = detrend(x[0].astype(float), axis=1)
    values /= np.maximum(values.std(axis=1, keepdims=True), 1e-8)
    p = abs(np.fft.rfft(values * np.hanning(values.shape[1]), axis=1)) ** 2
    f = np.fft.rfftfreq(values.shape[1], 1 / 30)
    motion = np.median(p[:, (f > .7) & (f <= 5)].sum(axis=1) / np.maximum(p[:, (f >= .1) & (f <= 5)].sum(axis=1), 1e-12))
    rates = [estimate_rates(x[..., start:start + 300])['mean_spectrum'] for start in (0, 300, 600)]
    return dict(zip(FEATURES, (float(motion), -concentration, -ratio, -agreement, float(np.std(rates)))))


def classify(estimate, reference):
    error = abs(estimate - reference)
    if error <= 5:
        return 'not_large'
    if abs(estimate - 2 * reference) <= 2:
        return 'near_double'
    if abs(estimate - reference / 2) <= 2:
        return 'near_half'
    return 'other_large'


def summary(rows):
    return {'count': len(rows), 'mae_bpm': float(np.mean([r['error'] for r in rows])) if rows else None,
            'large_errors': sum(r['error'] > 5 for r in rows),
            'within_2_bpm': float(np.mean([r['error'] <= 2 for r in rows])) if rows else None}


def select_gate(rows):
    # A good window is <=2 BPM; rejection budget 1%, overall retained >=80%.
    best = (0, 0, 'none', None)
    for feature in FEATURES:
        values = np.asarray([r['features'][feature] for r in rows])
        for threshold in np.unique(values):
            rejected = values > threshold
            good = np.asarray([r['error'] <= 2 for r in rows])
            if rejected.mean() > .2 or (np.sum(rejected & good) / max(1, good.sum())) > .01:
                continue
            removed = int(sum(rejected[i] and r['error'] > 5 for i, r in enumerate(rows)))
            key = (removed, -int(rejected.sum()))
            if key > best[:2]:
                best = (*key, feature, float(threshold))
    return {'feature': best[2], 'threshold': best[3], 'validation_large_removed': best[0]}


def main():
    base = Path('backend/modules/breathing_detection/outputs/respiratory_rate_baseline_v1')
    estimates = json.loads((base / 'window_estimates.json').read_text())['30']
    cache = load_wire_breath_cache('backend/modules/breathing_detection/outputs/wire_breath_cache', 1401)
    protocol = load_protocol(DEFAULT_PATH, cache.manifest['samples'])
    dataset = WireBreathDataset('data/WiReBreath', window_seconds=30, stride_seconds=30, normalize=False)
    rows = []
    for i, row in enumerate(estimates):
        sample = dataset[i]
        if row['sample_id'] != sample.sample_id:
            raise ValueError('Sample alignment mismatch')
        if not row['assessable'] or not row['estimates']:
            continue
        estimate = row['estimates']['mean_spectrum']
        reference = row['reference_bpm']
        rows.append({**row, 'estimate_bpm': estimate, 'error': abs(estimate - reference),
                     'error_type': classify(estimate, reference), 'features': features(sample.x)})
    folds = {}
    retained_all, rejected_all = [], []
    for name, fold in protocol['folds'].items():
        validation_recordings = {cache.manifest['samples'][i]['recording_id'] for i in fold['validation']}
        validation = [r for r in rows if r['recording_id'] in validation_recordings]
        assessment = [r for r in rows if r['participant_id'] == fold['held_out_subject']]
        gate = select_gate(validation)
        rejected = [r for r in assessment if gate['feature'] != 'none' and r['features'][gate['feature']] > gate['threshold']]
        rejected_ids = {r['sample_id'] for r in rejected}
        retained = [r for r in assessment if r['sample_id'] not in rejected_ids]
        retained_all.extend(retained)
        rejected_all.extend(rejected)
        folds[name] = {'gate': gate, 'before': summary(assessment), 'retained': summary(retained),
                       'rejected': summary(rejected), 'good_windows_rejected': sum(r['error'] <= 2 for r in rejected)}
    feature_comparisons = {}
    for feature in FEATURES:
        feature_comparisons[feature] = {}
        for group, selected in [('good', [r for r in rows if r['error'] <= 2]), ('large', [r for r in rows if r['error'] > 5])]:
            values = [r['features'][feature] for r in selected]
            feature_comparisons[feature][group] = dict(zip(('p10', 'median', 'p90'), map(float, np.quantile(values, [.1, .5, .9]))))
    large = [r for r in rows if r['error'] > 5]
    report = {
        'definitions': {'large_error_bpm': '>5', 'good_error_bpm': '<=2', 'harmonic_tolerance_bpm': 2,
                        'gate_selection': 'maximize inner-validation large errors removed; <=1% good rejected, >=80% retained; fixed feature order breaks ties',
                        'feature_sign': 'all features are risks: higher is less reliable; negative features invert intuitive confidence'},
        'overall': summary(rows), 'large_error_types': dict(Counter(r['error_type'] for r in large)),
        'large_by_environment': dict(Counter(r['environment'] for r in large)),
        'large_by_participant': dict(Counter(r['participant_id'] for r in large)),
        'large_by_recording': dict(Counter(r['recording_id'] for r in large)),
        'feature_comparisons_descriptive_only': feature_comparisons, 'folds': folds,
        'gate_outer_pooled': {'retained': summary(retained_all), 'rejected': summary(rejected_all),
                             'good_rejected': sum(r['error'] <= 2 for r in rejected_all),
                             'retained_of_quality_passing': len(retained_all) / len(rows),
                             'retained_of_all_467': len(retained_all) / 467},
        'large_cases': sorted(large, key=lambda r: -r['error']),
        'limitations': ['Near-half/double is numeric proximity, not proven harmonic causation.',
                       'Reference spectral peaks can themselves be unreliable; no independent annotated BPM.',
                       'Three-subwindow spread is within-window consistency, not continuous between-record tracking.',
                       'This is already-observed development cross-validation; no production gate promoted.'],
    }
    output = base / 'large_error_analysis.json'
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'large_cases'}, indent=2))


if __name__ == '__main__':
    main()
