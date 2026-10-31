"""Nested development evaluation of 10/30-second spectral rate baselines."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from backend.modules.breathing_detection.preprocessing.respiratory_rate import METHODS, estimate_rates, reference_rate
from backend.modules.breathing_detection.preprocessing.wire_breath import WireBreathDataset
from backend.modules.breathing_detection.preprocessing.wire_cache import load_wire_breath_cache
from backend.modules.breathing_detection.preprocessing.wire_protocol import DEFAULT_PATH, load_protocol


def metrics(rows, method):
    good = [row for row in rows if row['assessable'] and row['reference_bpm'] is not None and row['estimates'] is not None]
    errors = np.asarray([row['estimates'][method] - row['reference_bpm'] for row in good])
    by_record = defaultdict(list)
    for row, error in zip(good, errors):
        by_record[row['recording_id']].append(abs(float(error)))
    return {
        'windows': len(rows), 'assessable': len(good), 'coverage': len(good) / len(rows) if rows else None,
        'mae_bpm': float(np.mean(abs(errors))) if len(errors) else None,
        'rmse_bpm': float(np.sqrt(np.mean(errors ** 2))) if len(errors) else None,
        'bias_bpm': float(errors.mean()) if len(errors) else None,
        'within_2_bpm': float(np.mean(abs(errors) <= 2)) if len(errors) else None,
        'within_3_bpm': float(np.mean(abs(errors) <= 3)) if len(errors) else None,
        'record_macro_mae_bpm': float(np.mean([np.mean(v) for v in by_record.values()])) if by_record else None,
    }


def duration_diagnostics(all_rows, folds):
    selected_rows = {'10': [], '30': []}
    for name, fold in folds.items():
        subject = name.removeprefix('subject_')
        for duration in ('10', '30'):
            method = fold['by_duration'][duration]['method']
            for row in all_rows[duration]:
                if row['participant_id'] == subject:
                    estimates = {'selected': row['estimates'][method]} if row['estimates'] else None
                    selected_rows[duration].append({**row, 'estimates': estimates})
    common_segments = {r['sample_id'].split('@')[0] for r in selected_rows['30'] if r['assessable'] and r['estimates']}
    results = {}
    for duration, rows in selected_rows.items():
        common = [r for r in rows if r['sample_id'].split('@')[0] in common_segments]
        results[duration] = {
            'pooled': metrics(rows, 'selected'),
            'by_environment': {env: metrics([r for r in rows if r['environment'] == env], 'selected') for env in ('Quiet', 'Interference')},
            'matched_clean_segments': metrics(common, 'selected'),
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        report = json.loads((args.output_dir / 'metrics.json').read_text())
        rows = json.loads((args.output_dir / 'window_estimates.json').read_text())
        report['duration_diagnostics'] = duration_diagnostics(rows, report['folds'])
        (args.output_dir / 'metrics.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report['duration_diagnostics'], indent=2))
        return
    if (args.output_dir / 'config.json').exists():
        raise FileExistsError('Use a new experiment directory')
    cache = load_wire_breath_cache('backend/modules/breathing_detection/outputs/wire_breath_cache', 1401)
    protocol = load_protocol(DEFAULT_PATH, cache.manifest['samples'])
    config = {'protocol': str(DEFAULT_PATH), 'fingerprint': protocol['sample_fingerprint_sha256'],
              'methods': METHODS, 'windows_seconds': [10, 30], 'band_hz': [.1, .7],
              'reference': 'dominant detrended tapered spectrum of aligned respiratory sensor waveform',
              'selection': 'minimum inner-validation record-macro MAE; fixed tie order: 10s before 30s, method order as listed',
              'quality_max_gap_ms': 100, 'lstm_gate_used': False}
    (args.output_dir / 'config.json').write_text(json.dumps(config, indent=2))
    all_rows = {}
    for duration in (10, 30):
        dataset = WireBreathDataset('data/WiReBreath', window_seconds=duration, stride_seconds=duration, normalize=False)
        rows = []
        for index in range(len(dataset)):
            sample = dataset[index]
            estimates, reference = None, None
            flags = list(sample.quality_flags)
            if sample.assessable:
                try:
                    estimates = estimate_rates(sample.x)
                    reference = reference_rate(sample.respiratory_reference)
                except ValueError as error:
                    flags.append(str(error))
            rows.append({'sample_id': sample.sample_id, 'recording_id': sample.recording_id,
                         'participant_id': sample.metadata['participant_id'], 'environment': sample.metadata['environment'],
                         'room': sample.metadata['room'], 'posture': sample.metadata['posture'],
                         'assessable': sample.assessable and not flags, 'quality_flags': flags,
                         'reference_bpm': reference, 'estimates': estimates})
        all_rows[str(duration)] = rows
        print('Extracted', duration, 'seconds:', len(rows), 'windows', flush=True)
    folds = {}
    for name, fold in protocol['folds'].items():
        validation_recordings = {cache.manifest['samples'][i]['recording_id'] for i in fold['validation']}
        assessment_recordings = {cache.manifest['samples'][i]['recording_id'] for i in fold['assessment']}
        candidates = []
        for duration in (10, 30):
            validation = [r for r in all_rows[str(duration)] if r['recording_id'] in validation_recordings]
            for method in METHODS:
                result = metrics(validation, method)
                if result['record_macro_mae_bpm'] is None:
                    raise ValueError('No valid inner validation rates')
                candidates.append({'duration': duration, 'method': method, 'validation': result})
        selected = min(candidates, key=lambda item: item['validation']['record_macro_mae_bpm'])
        by_duration = {}
        for duration in (10, 30):
            chosen = min((c for c in candidates if c['duration'] == duration), key=lambda c: c['validation']['record_macro_mae_bpm'])
            rows = [r for r in all_rows[str(duration)] if r['recording_id'] in assessment_recordings]
            by_duration[str(duration)] = {
                'method': chosen['method'], 'assessment': metrics(rows, chosen['method']),
                'by_environment': {env: metrics([r for r in rows if r['environment'] == env], chosen['method']) for env in ('Quiet', 'Interference')},
            }
        folds[name] = {'candidates_inner_validation': candidates, 'selected': selected,
                       'assessment': by_duration[str(selected['duration'])], 'by_duration': by_duration}
    summary = {}
    for duration in ('10', '30'):
        values = [f['by_duration'][duration]['assessment']['mae_bpm'] for f in folds.values()]
        summary[duration] = {'person_macro_mae_bpm': float(np.mean(values)), 'worst_person_mae_bpm': float(max(values))}
    report = {'config': config, 'folds': folds, 'duration_summary': summary,
              'duration_diagnostics': duration_diagnostics(all_rows, folds),
              'limitations': ['Reference BPM is a spectral proxy derived from a respiratory waveform, not independently annotated rate.',
                             'Zero padding refines peak search, not the physical frequency resolution.',
                             'Different durations have different reference intervals and quality coverage; report both.',
                             'Already-observed development data; no clinical or independent final-test claim.']}
    (args.output_dir / 'window_estimates.json').write_text(json.dumps(all_rows, indent=2))
    (args.output_dir / 'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'summary': summary, 'folds': {k: {'selected': v['selected'], 'assessment': v['assessment']} for k, v in folds.items()}}, indent=2))


if __name__ == '__main__':
    main()
