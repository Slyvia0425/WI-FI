"""Nested CSI-only rate withholding experiment after boundary correction."""
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from backend.modules.breathing_detection.preprocessing.respiratory_rate import spectra
from backend.modules.breathing_detection.preprocessing.wire_breath import WireBreathDataset
from backend.modules.breathing_detection.preprocessing.wire_cache import load_wire_breath_cache
from backend.modules.breathing_detection.preprocessing.wire_protocol import DEFAULT_PATH, load_protocol
from .analyze_rate_errors import classify
from .evaluate_local_rate_peaks import evaluate


def confidence_features(frequencies, power, bpm):
    mean = power.mean(axis=0)
    hz = bpm / 60
    main = int(np.argmin(abs(frequencies - hz)))
    peaks, _ = find_peaks(mean)
    competitors = [mean[p] for p in peaks if abs(frequencies[p] - hz) > 2 / 30]
    return {
        'concentration': float(mean[abs(frequencies - hz) <= 1 / 30].sum() / max(mean.sum(), 1e-12)),
        'agreement': float(np.mean(abs(frequencies[np.argmax(power, axis=1)] - hz) <= 1 / 30)),
        'peak_ratio': float(mean[main] / max(max(competitors, default=0), 1e-12)),
    }


def reject(features, rule):
    if rule is None:
        return False
    weak = [features[k] < rule[k] for k in ('concentration', 'agreement', 'peak_ratio')]
    return sum(weak) >= rule['required_weak']


def select_rule(rows, features, method, rules):
    good = np.asarray([abs(r['predictions'][method] - r['reference_bpm']) <= 2 for r in rows])
    large = np.asarray([abs(r['predictions'][method] - r['reference_bpm']) > 5 for r in rows])
    best_rule, best_key = None, (0, 0, 0)
    for rule in rules:
        rejected = np.asarray([reject(features[r['sample_id']][method], rule) for r in rows])
        harmed = int(np.sum(rejected & good))
        if rejected.mean() > .15 or harmed / max(1, good.sum()) > .01:
            continue
        key = (int(np.sum(rejected & large)), -harmed, -int(rejected.sum()))
        if key > best_key:
            best_rule, best_key = rule, key
    return best_rule, {'large_removed': best_key[0], 'good_rejected': -best_key[1], 'total_rejected': -best_key[2]}


def main():
    root = Path('backend/modules/breathing_detection/outputs')
    source = root / 'respiratory_rate_boundary_only_v1'
    output = root / 'respiratory_rate_reliability_v1'
    output.mkdir(exist_ok=True)
    if (output / 'config.json').exists():
        raise FileExistsError('Preserve existing experiment')
    original = json.loads((source / 'metrics.json').read_text())
    rows = json.loads((source / 'window_predictions.json').read_text())
    cache = load_wire_breath_cache(root / 'wire_breath_cache', 1401)
    protocol = load_protocol(DEFAULT_PATH, cache.manifest['samples'])
    if original['config']['protocol_sha256'] != protocol['sample_fingerprint_sha256']:
        raise ValueError('Source fingerprint mismatch')
    rules = [dict(concentration=c, agreement=a, peak_ratio=p, required_weak=n)
             for c, a, p, n in itertools.product((.1, .15, .2, .25), (.1, .2, .3), (1.1, 1.3, 1.5), (2, 3))]
    config = {'protocol_sha256': protocol['sample_fingerprint_sha256'], 'rules': rules,
              'validation_budget': 'reject <=15% quality-passing; reject <=1% currently accurate estimates',
              'selection': 'maximize >5 BPM errors rejected, then minimize good and total rejections; no gate fallback',
              'fold_estimator': 'use that fold boundary estimator for BOTH inner validation and outer assessment',
              'application_enabled': False}
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    by_id = {r['sample_id']: r for r in rows}
    dataset = WireBreathDataset('data/WiReBreath', window_seconds=30, stride_seconds=30, normalize=False)
    methods = sorted({f['selected'] for f in original['folds'].values()})
    features = {}
    for i in range(len(dataset)):
        sample = dataset[i]
        if sample.sample_id not in by_id:
            continue
        f, power = spectra(sample.x[0])
        row = by_id[sample.sample_id]
        features[sample.sample_id] = {m: confidence_features(f, power, row['predictions'][m]) for m in methods}
    folds, pooled = {}, []
    for name, fold in protocol['folds'].items():
        method = original['folds'][name]['selected']
        validation_ids = {cache.manifest['samples'][i]['recording_id'] for i in fold['validation']}
        validation = [r for r in rows if r['recording_id'] in validation_ids]
        rule, selection = select_rule(validation, features, method, rules)
        assessment = [r for r in rows if r['participant_id'] == fold['held_out_subject']]
        outcomes = []
        for row in assessment:
            withheld = reject(features[row['sample_id']][method], rule)
            prediction = row['predictions'][method]
            outcomes.append({**row, 'predictions': {**row['predictions'], 'current': prediction,
                              'withheld': None if withheld else prediction},
                             'confidence_features': features[row['sample_id']][method],
                             'rate_withheld': withheld,
                             'rate_reason': 'weak_spectral_evidence' if withheld else None,
                             'breathing_status_changed': False})
        pooled.extend(outcomes)
        before, after = evaluate(outcomes, 'current'), evaluate(outcomes, 'withheld')
        # Evaluate harm relative to the current rate estimator, not the older baseline.
        good = [r for r in outcomes if abs(r['predictions']['current'] - r['reference_bpm']) <= 2]
        harmed = sum(r['rate_withheld'] for r in good)
        folds[name] = {'estimator': method, 'rule': rule, 'inner_selection': selection,
                       'before': before, 'after': after, 'current_good': len(good), 'current_good_rejected': harmed}
    good = [r for r in pooled if abs(r['predictions']['current'] - r['reference_bpm']) <= 2]
    harmed = sum(r['rate_withheld'] for r in good)
    remaining = [r for r in pooled if abs(r['predictions']['current'] - r['reference_bpm']) > 5]
    rejected_large = sum(r['rate_withheld'] for r in remaining)
    report = {'config': config, 'folds': folds,
              'before': evaluate(pooled, 'current'), 'after': evaluate(pooled, 'withheld'),
              'current_good': len(good), 'current_good_rejected': harmed,
              'current_good_rejection_rate': harmed / len(good),
              'large_rejected': rejected_large, 'large_total': len(remaining),
              'total_window_coverage': sum(not r['rate_withheld'] for r in pooled) / 467,
              'remaining22_types': dict(Counter(classify(r['predictions']['current'], r['reference_bpm']) for r in remaining)),
              'remaining22_by_environment': dict(Counter(r['environment'] for r in remaining)),
              'outer_safety_pass': harmed / len(good) <= .01 and all(f['current_good_rejected'] / max(1, f['current_good']) <= .01 for f in folds.values()),
              'limitations': ['Already-observed development data; no independent final-test claim.',
                             'No threshold deployed; missing rate must not alter breathing evidence status.',
                             'Shared inner validation chooses estimator and gate; estimates of generalization are exploratory.']}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2))
    (output / 'window_results.json').write_text(json.dumps(pooled, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ('folds', 'config')}, indent=2))
    print('Folds:', json.dumps({k: {'rule': f['rule'], 'rejected_good': f['current_good_rejected'], 'retained': f['after']['retained']} for k, f in folds.items()}))


if __name__ == '__main__':
    main()
