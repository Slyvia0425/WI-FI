"""Conservative boundary-only correction using fixed nested development folds."""
import json
from pathlib import Path

import numpy as np

from backend.modules.breathing_detection.preprocessing.rate_peaks import conservative_boundary_rate, search_endpoints
from backend.modules.breathing_detection.preprocessing.wire_cache import load_wire_breath_cache
from backend.modules.breathing_detection.preprocessing.wire_protocol import DEFAULT_PATH, load_protocol
from .evaluate_local_rate_peaks import evaluate


def main():
    root = Path('backend/modules/breathing_detection/outputs')
    source = root / 'respiratory_rate_local_peaks_v1'
    output = root / 'respiratory_rate_boundary_only_v1'
    cache = load_wire_breath_cache(root / 'wire_breath_cache', 1401)
    protocol = load_protocol(DEFAULT_PATH, cache.manifest['samples'])
    source_config = json.loads((source / 'config.json').read_text())
    if source_config['protocol_sha256'] != protocol['sample_fingerprint_sha256']:
        raise ValueError('Candidate features use a different cache fingerprint')
    rows = json.loads((source / 'window_predictions.json').read_text())
    expected = {r['sample_id'] for r in cache.manifest['samples'] if r['sample_id'].endswith('@0.000s')}
    if len({r['sample_id'] for r in rows}) != len(rows) or not {r['sample_id'] for r in rows} <= expected:
        raise ValueError('Unexpected/duplicate 30-second source IDs')
    configs = {f'boundary_prom{prom}': prom for prom in (0., .05, .1, .2)}
    config = {'protocol_sha256': protocol['sample_fingerprint_sha256'],
              'rule': 'exact endpoint only; highest local peak; preserve interior and no-valid-peak outputs',
              'endpoints_bpm': search_endpoints(), 'minimum_prominence_candidates': configs,
              'selection': 'inner record-macro MAE; no baseline-good harm allowed; baseline eligible fallback',
              'source': str(source), 'application_enabled': False}
    output.mkdir(exist_ok=True)
    if (output / 'config.json').exists():
        raise FileExistsError('Preserve configured run; use a new version')
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    for row in rows:
        baseline = row['predictions']['baseline']
        row['predictions'] = {'baseline': baseline, **{
            name: conservative_boundary_rate(baseline, row['peaks'], prom) for name, prom in configs.items()}}
    folds, selected_rows = {}, []
    for name, fold in protocol['folds'].items():
        validation_records = {cache.manifest['samples'][i]['recording_id'] for i in fold['validation']}
        validation = [r for r in rows if r['recording_id'] in validation_records]
        candidates = {method: evaluate(validation, method) for method in ('baseline', *configs)}
        eligible = [method for method, m in candidates.items() if m['good_harm_rate'] == 0]
        chosen = min(eligible, key=lambda m: candidates[m]['record_macro_mae_bpm'])
        assessment = [r for r in rows if r['participant_id'] == fold['held_out_subject']]
        selected_rows.extend([{**r, 'predictions': {**r['predictions'], 'selected': r['predictions'][chosen]}, 'chosen_method': chosen} for r in assessment])
        folds[name] = {'selected': chosen, 'validation_candidates': candidates,
                       'assessment_baseline': evaluate(assessment, 'baseline'), 'assessment_selected': evaluate(assessment, chosen)}
    changed = [r for r in selected_rows if abs(r['predictions']['selected'] - r['predictions']['baseline']) > 1e-6]
    new_large = [r['sample_id'] for r in changed if r['baseline_error'] <= 5 and abs(r['predictions']['selected'] - r['reference_bpm']) > 5]
    low, high = search_endpoints()
    interior_changes = [r['sample_id'] for r in changed if abs(r['predictions']['baseline'] - low) > 1e-6 and abs(r['predictions']['baseline'] - high) > 1e-6]
    assert not interior_changes
    low_rate = [r for r in selected_rows if r['reference_bpm'] < 10]
    report = {'config': config, 'folds': folds,
              'baseline': evaluate(rows, 'baseline'), 'selected': evaluate(selected_rows, 'selected'),
              'fixed_boundary_zero_prominence_control': evaluate(rows, 'boundary_prom0.0'),
              'person_macro_mae_bpm': float(np.mean([f['assessment_selected']['mae_bpm'] for f in folds.values()])),
              'total_window_coverage': len(selected_rows) / 467,
              'changed_windows': len(changed), 'new_large_error_ids': new_large,
              'interior_changed_ids': interior_changes,
              'reference_below10': {'baseline': evaluate(low_rate, 'baseline'), 'selected': evaluate(low_rate, 'selected')},
              'by_environment': {env: evaluate([r for r in selected_rows if r['environment'] == env], 'selected') for env in ('Quiet', 'Interference')},
              'limitations': ['Reuse of CSI-only spectra; previous learned fold selections ignored.',
                             'Historically observed development data; no independent final validation.',
                             'Reference is waveform-derived; no uncertainty gate or physical cause proven.']}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2))
    (output / 'window_predictions.json').write_text(json.dumps(selected_rows, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'folds'}, indent=2))
    print('Selections:', {name: fold['selected'] for name, fold in folds.items()})


if __name__ == '__main__':
    main()
