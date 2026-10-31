"""One predeclared carrier-supported correction vs frozen unified baseline."""
import json
from pathlib import Path

from ..preprocessing.peak_support import RULE_ID, supported_competitor
from ..preprocessing.wire_cache import load_wire_breath_cache
from ..preprocessing.wire_protocol import DEFAULT_PATH, load_protocol
from .evaluate_spike_preprocessing import ROOT, digest, select_preprocessing
from .evaluate_local_rate_peaks import evaluate


def main():
    output = ROOT/'peak_support_v1'
    if output.exists():
        raise FileExistsError('Preserve frozen supported-competitor experiment')
    source = ROOT/'peak_evidence_v1/window_evidence.json'
    audit_config = json.loads((ROOT/'peak_evidence_v1/config.json').read_text())
    protected = audit_config['protected_artifact_hashes']
    if any(digest(path) != sha for path, sha in protected.items()):
        raise ValueError('Protected artifact changed before evaluation')
    cache = load_wire_breath_cache(ROOT/'wire_breath_cache', 1401)
    protocol = load_protocol(DEFAULT_PATH, cache.manifest['samples'])
    config = {'rule_id': RULE_ID, 'application_enabled': False,
        'source_sha256': digest(source), 'evaluation_sha256': digest(__file__),
        'implementation_sha256': digest(Path(__file__).parents[1]/'preprocessing/peak_support.py'),
        'candidate_protocol_sha256': digest(Path(__file__).parents[1]/'PEAK_SUPPORT_CANDIDATE_PROTOCOL.md'),
        'protocol_sample_fingerprint_sha256': protocol['sample_fingerprint_sha256'],
        'protected_artifact_hashes': protected, 'minimum_height': .5, 'tolerance_bpm': 2.,
        'selection': 'inner record-macro MAE; full coverage and zero accurate harm; ties baseline/fewer changes',
        'support': 'zero harm, full coverage, >=25% fewer large errors, no person MAE worsening',
        'limitations': ['Already observed retrospective development data, not independent final assessment.',
                        'Carrier support is correlated spectral evidence, not motion identification or calibrated confidence.',
                        'No preprocessing or rejection; no alternate threshold after scoring.']}
    output.mkdir()
    (output/'config.json').write_text(json.dumps(config, indent=2))
    evidence_rows = json.loads(source.read_text())
    rows = []
    for source_row in evidence_rows:
        baseline = source_row['predictions']['baseline']
        prediction = supported_competitor(source_row['evidence']['baseline'], baseline)
        row = {k: source_row[k] for k in ('sample_id', 'recording_id', 'participant_id', 'reference_bpm', 'baseline_error')}
        row.update(predictions={'baseline': baseline, RULE_ID: prediction['bpm']}, decision=prediction)
        rows.append(row)
    assert len(rows) == 394 and len({r['sample_id'] for r in rows}) == 394
    methods = ['baseline', RULE_ID]
    folds, pooled = {}, []
    for name, fold in protocol['folds'].items():
        records = {part: {cache.manifest['samples'][i]['recording_id'] for i in fold[part]}
                   for part in ('train', 'validation', 'assessment')}
        if records['train'] & (records['validation'] | records['assessment']) or records['validation'] & records['assessment']:
            raise ValueError('Recording split leakage')
        validation = [r for r in rows if r['recording_id'] in records['validation']]
        assessment = [r for r in rows if r['recording_id'] in records['assessment']]
        assert all(r['participant_id'] != fold['held_out_subject'] for r in validation)
        assert all(r['participant_id'] == fold['held_out_subject'] for r in assessment)
        chosen, inner = select_preprocessing(validation, methods)
        outcomes = [{**r, 'predictions': {**r['predictions'], 'selected': r['predictions'][chosen]},
                     'selected_method': chosen} for r in assessment]
        pooled.extend(outcomes)
        folds[name] = {'chosen': chosen, 'inner_candidates': inner,
            'validation_ids': [r['sample_id'] for r in validation],
            'assessment_ids': [r['sample_id'] for r in assessment],
            'baseline': evaluate(outcomes, 'baseline'), 'selected': evaluate(outcomes, 'selected'),
            'fixed_candidate': evaluate(outcomes, RULE_ID)}
    assert len(pooled) == len(rows) == len({r['sample_id'] for r in pooled})
    baseline, selected, fixed = (evaluate(pooled, method) for method in ('baseline', 'selected', RULE_ID))
    safety = selected['retained'] == len(rows) and selected['good_harm_rate'] == 0
    reduction = (baseline['large_errors']-selected['large_errors'])/baseline['large_errors']
    nonworse = all(f['selected']['mae_bpm'] <= f['baseline']['mae_bpm'] for f in folds.values())
    changed = [r for r in rows if r['decision']['changed']]
    summary = {'baseline': baseline, 'selected': selected, 'fixed_candidate': fixed, 'folds': folds,
        'fixed_changed_windows': len(changed),
        'fixed_new_large_errors': sum(r['baseline_error'] <= 5 and abs(r['predictions'][RULE_ID]-r['reference_bpm']) > 5 for r in rows),
        'selected_changed_windows': sum(r['predictions']['selected'] != r['predictions']['baseline'] for r in pooled),
        'outer_safety_pass': safety, 'large_error_reduction_fraction': reduction,
        'per_person_mae_nonworse': nonworse, 'development_support_pass': safety and reduction >= .25 and nonworse,
        'total_window_coverage': selected['retained']/467, 'application_enabled': False}
    if any(digest(path) != sha for path, sha in protected.items()):
        raise ValueError('Protected artifact changed')
    assert digest(source) == config['source_sha256']
    (output/'window_predictions.json').write_text(json.dumps(pooled, indent=2))
    (output/'changed_cases.json').write_text(json.dumps(changed, indent=2))
    (output/'metrics.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'folds'}, indent=2))
    print('Selections', {k: v['chosen'] for k, v in folds.items()})


if __name__ == '__main__':
    main()
