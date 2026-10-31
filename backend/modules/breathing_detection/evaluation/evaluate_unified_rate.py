"""Full fixed-estimator development evaluation, no refitting or fold switching."""
import json
from pathlib import Path
import hashlib
import numpy as np

from ..preprocessing.unified_rate import estimate_unified_rate, ESTIMATOR_SPEC
from ..preprocessing.wire_breath import WireBreathDataset
from .evaluate_local_rate_peaks import evaluate

ROOT = Path('backend/modules/breathing_detection/outputs')


def main():
    output = ROOT / 'respiratory_rate_unified_v1'
    if output.exists():
        raise FileExistsError('Preserve fixed-estimator run; use a reviewed new version')
    source = ROOT / 'respiratory_rate_boundary_only_v1/window_predictions.json'
    original = json.loads(source.read_text())
    by_id = {r['sample_id']: r for r in original}
    config = {'estimator': ESTIMATOR_SPEC, 'selection': 'fixed boundary_prom0.0 after retrospective development comparison; no fitting in this run',
              'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'implementation_sha256': hashlib.sha256((Path(__file__).parents[1] / 'preprocessing/unified_rate.py').read_bytes()).hexdigest(),
              'limitations': ['Already-observed development data; post-comparison choice, not independent or nested-CV performance.',
                              'No subject-specific parameter selection. Reliability gates disabled.',
                              'Reference rates are waveform-derived. No clinical or long-duration reliability claim.']}
    output.mkdir()
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    dataset = WireBreathDataset('data/WiReBreath', window_seconds=30, stride_seconds=30, normalize=False)
    rows, excluded = [], []
    for index in range(len(dataset)):
        sample = dataset[index]
        if not sample.assessable:
            excluded.append({'sample_id': sample.sample_id, 'reason': list(sample.quality_flags)})
            continue
        if sample.sample_id not in by_id:
            raise ValueError('Quality-passing sample missing from existing reference evaluation')
        source_row = by_id[sample.sample_id]
        estimate = estimate_unified_rate(sample.x)
        # Verify shared implementation reproduces the declared fixed control.
        if not np.isclose(estimate['bpm'], source_row['predictions']['boundary_prom0.0'], atol=1e-9, rtol=0):
            raise ValueError('Unified estimator differs from frozen fixed-rule control')
        if not np.isclose(estimate['baseline_bpm'], source_row['predictions']['baseline'], atol=1e-9, rtol=0):
            raise ValueError('Baseline representation mismatch')
        rows.append({**source_row, 'estimate': estimate,
                     'predictions': {**source_row['predictions'], 'unified': estimate['bpm']}})
    if len(rows) != len(original) or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Fixed evaluation must cover the same valid reference set exactly once')
    people = {subject: {'baseline': evaluate([r for r in rows if r['participant_id'] == subject], 'baseline'),
                        'unified': evaluate([r for r in rows if r['participant_id'] == subject], 'unified')}
              for subject in sorted({r['participant_id'] for r in rows})}
    baseline, unified = evaluate(rows, 'baseline'), evaluate(rows, 'unified')
    changed = [r for r in rows if r['estimate']['correction_applied']]
    report = {'config': config, 'total_segments': len(dataset), 'quality_excluded': len(excluded),
              'baseline': baseline, 'unified': unified,
              'person_macro_mae_bpm': float(np.mean([p['unified']['mae_bpm'] for p in people.values()])),
              'total_window_coverage': len(rows)/len(dataset), 'by_person': people,
              'by_environment': {env: evaluate([r for r in rows if r['environment'] == env], 'unified') for env in ('Quiet','Interference')},
              'changed_windows': len(changed),
              'new_large_error_ids': [r['sample_id'] for r in changed if r['baseline_error'] <= 5 and abs(r['predictions']['unified']-r['reference_bpm']) > 5],
              'fixed_control_match_all': True, 'reliability_gate_enabled': False}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2))
    (output / 'window_predictions.json').write_text(json.dumps(rows, indent=2))
    (output / 'excluded_windows.json').write_text(json.dumps(excluded, indent=2))
    print(json.dumps({k: v for k,v in report.items() if k not in ('config','by_person','by_environment')}, indent=2))


if __name__ == '__main__':
    main()
