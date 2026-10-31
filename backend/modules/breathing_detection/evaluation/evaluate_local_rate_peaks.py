"""Fixed nested comparison of bounded argmax and genuine local spectral peaks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from backend.modules.breathing_detection.preprocessing.rate_peaks import local_candidates, choose_peak
from backend.modules.breathing_detection.preprocessing.wire_breath import WireBreathDataset
from backend.modules.breathing_detection.preprocessing.wire_cache import load_wire_breath_cache
from backend.modules.breathing_detection.preprocessing.wire_protocol import DEFAULT_PATH, load_protocol


def evaluate(rows, method):
    kept = [r for r in rows if r['predictions'][method] is not None]
    errors = [abs(r['predictions'][method] - r['reference_bpm']) for r in kept]
    good = [r for r in rows if r['baseline_error'] <= 2]
    rejected_good = sum(r['predictions'][method] is None for r in good)
    spoiled_good = sum(r['predictions'][method] is not None and abs(r['predictions'][method] - r['reference_bpm']) > 2 for r in good)
    records = {}
    for row, error in zip(kept, errors):
        records.setdefault(row['recording_id'], []).append(error)
    return {'quality_passing_windows': len(rows), 'retained': len(kept),
            'coverage_of_quality_passing': len(kept) / len(rows) if rows else None,
            'mae_bpm': float(np.mean(errors)) if errors else None,
            'rmse_bpm': float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
            'record_macro_mae_bpm': float(np.mean([np.mean(v) for v in records.values()])) if records else None,
            'large_errors': sum(e > 5 for e in errors), 'within_2_bpm': float(np.mean(np.asarray(errors) <= 2)) if errors else None,
            'baseline_good': len(good), 'good_rejected': rejected_good, 'good_spoiled': spoiled_good,
            'good_harm_rate': (rejected_good + spoiled_good) / len(good) if good else 0.,
            'large_errors_corrected_to_within2': sum(r['baseline_error'] > 5 and r['predictions'][method] is not None and abs(r['predictions'][method] - r['reference_bpm']) <= 2 for r in rows)}


def main():
    base = Path('backend/modules/breathing_detection/outputs/respiratory_rate_baseline_v1')
    output = Path('backend/modules/breathing_detection/outputs/respiratory_rate_local_peaks_v1')
    output.mkdir(exist_ok=True)
    if (output / 'config.json').exists():
        raise FileExistsError('Preserve existing experiment; use a new version')
    cache = load_wire_breath_cache('backend/modules/breathing_detection/outputs/wire_breath_cache', 1401)
    protocol = load_protocol(DEFAULT_PATH, cache.manifest['samples'])
    configs = {f'{ranking}_prom{prom}': (ranking, prom) for ranking in ('height', 'prominence') for prom in (0., .05, .1, .2)}
    config = {'protocol_sha256': protocol['sample_fingerprint_sha256'], 'candidates': configs,
              'selection': 'inner record-macro MAE, retain >=90% quality-passing, harm <=1% baseline-good; baseline is eligible fallback',
              'good_harm': 'rejection OR corrected rate outside 2 BPM for originally accurate samples',
              'spectrum_context_hz': [.05, 1.], 'allowed_rate_hz': [.1, .7], 'window_seconds': 30,
              'promotion': 'diagnostic experiment only; inspect outer safety before any integration'}
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    source = json.loads((base / 'window_estimates.json').read_text())['30']
    dataset = WireBreathDataset('data/WiReBreath', window_seconds=30, stride_seconds=30, normalize=False)
    rows = []
    for i, row in enumerate(source):
        if not row['assessable'] or not row['estimates']:
            continue
        sample = dataset[i]
        if row['sample_id'] != sample.sample_id:
            raise ValueError('Sample order mismatch')
        peaks = local_candidates(sample.x)
        predictions = {'baseline': row['estimates']['mean_spectrum']}
        predictions.update({name: choose_peak(peaks, ranking, prom) for name, (ranking, prom) in configs.items()})
        rows.append({**row, 'peaks': peaks, 'predictions': predictions,
                     'baseline_error': abs(predictions['baseline'] - row['reference_bpm'])})
    folds = {}
    selected_rows = []
    for name, fold in protocol['folds'].items():
        validation_records = {cache.manifest['samples'][i]['recording_id'] for i in fold['validation']}
        validation = [r for r in rows if r['recording_id'] in validation_records]
        candidates = {method: evaluate(validation, method) for method in ('baseline', *configs)}
        eligible = [method for method, m in candidates.items() if m['coverage_of_quality_passing'] >= .9 and m['good_harm_rate'] <= .01]
        chosen = min(eligible, key=lambda method: candidates[method]['record_macro_mae_bpm'])
        assessment = [r for r in rows if r['participant_id'] == fold['held_out_subject']]
        selected_rows.extend([{**r, 'predictions': {**r['predictions'], 'selected': r['predictions'][chosen]}, 'chosen_method': chosen} for r in assessment])
        folds[name] = {'selected_method': chosen, 'inner_candidates': candidates,
                       'assessment_baseline': evaluate(assessment, 'baseline'), 'assessment_selected': evaluate(assessment, chosen)}
    report = {'config': config, 'folds': folds, 'pooled_baseline': evaluate(rows, 'baseline'),
              'pooled_selected': evaluate(selected_rows, 'selected'),
              'fixed_local_height_control': evaluate(rows, 'height_prom0.0'),
              'by_environment': {env: evaluate([r for r in selected_rows if r['environment'] == env], 'selected') for env in ('Quiet', 'Interference')},
              'person_macro_mae_bpm': float(np.mean([f['assessment_selected']['mae_bpm'] for f in folds.values()])),
              'total_window_coverage': sum(r['predictions']['selected'] is not None for r in selected_rows) / len(source),
              'limitations': ['Historically observed development data; outer results are not independent final validation.',
                             'Reference remains waveform-derived spectral rate.', 'No model or production estimator replaced.']}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2))
    (output / 'window_predictions.json').write_text(json.dumps(selected_rows, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'folds'}, indent=2))
    print('Selections:', {name: fold['selected_method'] for name, fold in folds.items()})


if __name__ == '__main__':
    main()
