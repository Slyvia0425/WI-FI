"""Prespecified grouped development experiment: temporal/harmonic BPM withholding."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ..preprocessing.rate_temporal_reliability import (
    temporal_context, candidate_features, reliability_rules, withhold_rate,
    SUBWINDOW_SECONDS, SUBWINDOW_STARTS, TOLERANCE_BPM,
)
from ..preprocessing.wire_breath import WireBreathDataset
from ..preprocessing.wire_cache import load_wire_breath_cache
from ..preprocessing.wire_protocol import DEFAULT_PATH, load_protocol
from .analyze_rate_errors import classify

ROOT = Path('backend/modules/breathing_detection/outputs')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(rows, method, withheld=None):
    withheld = np.zeros(len(rows), dtype=bool) if withheld is None else np.asarray(withheld, dtype=bool)
    if len(rows) != len(withheld) or not rows:
        raise ValueError('Require nonempty aligned rows and withholding mask')
    error = np.asarray([abs(r['predictions'][method] - r['reference_bpm']) for r in rows])
    good, large = error <= 2, error > 5
    kept = error[~withheld]
    record_errors = {}
    for row, err, rejected in zip(rows, error, withheld):
        if not rejected:
            record_errors.setdefault(row['recording_id'], []).append(float(err))
    return {'quality_passing': len(rows), 'retained': int((~withheld).sum()),
            'coverage': float((~withheld).mean()), 'withheld': int(withheld.sum()),
            'mae_bpm': float(kept.mean()) if len(kept) else None,
            'rmse_bpm': float(np.sqrt(np.mean(kept**2))) if len(kept) else None,
            'record_macro_mae_bpm': float(np.mean([np.mean(v) for v in record_errors.values()])) if record_errors else None,
            'within2_retained': float((kept <= 2).mean()) if len(kept) else None,
            'current_good': int(good.sum()), 'good_withheld': int((good & withheld).sum()),
            'good_withholding_rate': float((good & withheld).sum()/good.sum()) if good.any() else 0.,
            'large_total': int(large.sum()), 'large_withheld': int((large & withheld).sum()),
            'large_retained': int((large & ~withheld).sum()),
            'large_interception_rate': float((large & withheld).sum()/large.sum()) if large.any() else None}


def budget_pass(rows, method, mask):
    groups = [rows] + [[r for r in rows if r['participant_id'] == subject]
                       for subject in sorted({r['participant_id'] for r in rows})]
    by_id = {r['sample_id']: bool(value) for r, value in zip(rows, mask)}
    return all((m := metrics(group, method, [by_id[r['sample_id']] for r in group]))['good_withholding_rate'] <= .01
               and m['withheld']/len(group) <= .15 for group in groups)


def select_rule(rows, features, method, rules):
    best, best_key, audit = None, (0, 0, 0), []
    for rule in rules:
        mask = [withhold_rate(features[r['sample_id']][method], rule) for r in rows]
        m = metrics(rows, method, mask)
        eligible = budget_pass(rows, method, mask)
        audit.append({'rule': rule, 'eligible': eligible, 'metrics': m})
        key = (m['large_withheld'], -m['good_withheld'], -m['withheld'])
        if eligible and key > best_key:
            best, best_key = rule, key
    return best, audit


def score_outcomes(outcomes, withheld=False):
    return metrics(outcomes, 'current', [r['rate_withheld'] if withheld else False for r in outcomes])


def main():
    source = ROOT / 'respiratory_rate_boundary_only_v1'
    output = ROOT / 'respiratory_rate_reliability_stage2_v1'
    if output.exists():
        raise FileExistsError('Preserve experiment directory; use a reviewed new version')
    cache = load_wire_breath_cache(ROOT / 'wire_breath_cache', 1401)
    protocol = load_protocol(DEFAULT_PATH, cache.manifest['samples'])
    original = json.loads((source / 'metrics.json').read_text())
    rows = json.loads((source / 'window_predictions.json').read_text())
    if original['config']['protocol_sha256'] != protocol['sample_fingerprint_sha256']:
        raise ValueError('Protocol fingerprint mismatch')
    all_samples = {r['sample_id']: r for r in cache.manifest['samples'] if r['sample_id'].endswith('@0.000s')}
    if len(rows) != len({r['sample_id'] for r in rows}) or not {r['sample_id'] for r in rows} <= set(all_samples):
        raise ValueError('Unexpected or duplicate source rows')
    for row in rows:
        meta = all_samples[row['sample_id']]
        if any(row[key] != meta[key] for key in ('participant_id', 'recording_id')):
            raise ValueError('Source grouping metadata mismatch')
    rules = reliability_rules()
    config = {'protocol_sha256': protocol['sample_fingerprint_sha256'],
              'source_predictions_sha256': sha256(source / 'window_predictions.json'),
              'source_metrics_sha256': sha256(source / 'metrics.json'),
              'feature_code_sha256': sha256(Path(__file__).parents[1] / 'preprocessing/rate_temporal_reliability.py'),
              'evaluation_code_sha256': sha256(Path(__file__)),
              'subwindow_seconds': SUBWINDOW_SECONDS, 'subwindow_starts_seconds': SUBWINDOW_STARTS,
              'subwindow_tolerance_bpm': TOLERANCE_BPM, 'full_harmonic_tolerance_bpm': 2.,
              'harmonic_candidates_hz': [.1, .7], 'rules': rules,
              'selection': 'inner validation only: maximize >5 BPM errors withheld; ties minimize <=2 BPM good withholding, then total withholding; no-gate fallback',
              'budget': 'pooled AND each person: <=1% good withholding and <=15% total withholding, inner and outer',
              'estimator': 'same previously inner-selected boundary estimator used for BOTH inner validation and outer assessment; not a unified deployment estimator',
              'promotion': 'outer safety passes AND >=25% large errors intercepted AND retained MAE does not worsen; development support only, not automatic activation',
              'application_enabled': False, 'reference_access': 'evaluation/inner rule selection only; feature extraction accepts CSI and candidate BPM only',
              'window_semantics': 'three overlapping 20s subdivisions INSIDE one 30s file; never across independent files',
              'limitations': ['Already-observed development data; not new unbiased validation.',
                              'Estimator and reliability rule reuse inner validation; exploratory generalization.',
                              'Subwindows overlap and are not independent replicates; zero padding does not improve physical frequency resolution.',
                              'Small per-person denominators make the 1% harm budget discrete.',
                              'Feature absence preserves the current estimator for this offline experiment, not a reliable status.']}
    output.mkdir()
    # Persist specification before accessing real CSI features or outer errors.
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    by_id = {r['sample_id']: r for r in rows}
    methods = sorted({fold['selected'] for fold in original['folds'].values()})
    features = {}
    dataset = WireBreathDataset('data/WiReBreath', window_seconds=30, stride_seconds=30, normalize=False)
    for index in range(len(dataset)):
        sample = dataset[index]
        if sample.sample_id not in by_id:
            continue
        row = by_id[sample.sample_id]
        try:
            context = temporal_context(sample.x)
            features[sample.sample_id] = {method: candidate_features(context, row['predictions'][method]) for method in methods}
        except ValueError as error:
            features[sample.sample_id] = {method: dict(available=False, reason=str(error)) for method in methods}
        if len(features) % 100 == 0:
            print('Extracted', len(features), 'CSI-only feature sets', flush=True)
    if set(features) != set(by_id):
        raise ValueError('Incomplete feature coverage')
    (output / 'features.json').write_text(json.dumps(features, indent=2))
    folds, pooled, controls = {}, [], {rule['name']: [] for rule in rules}
    for name, fold in protocol['folds'].items():
        method = original['folds'][name]['selected']
        validation_records = {cache.manifest['samples'][i]['recording_id'] for i in fold['validation']}
        assessment_records = {cache.manifest['samples'][i]['recording_id'] for i in fold['assessment']}
        if validation_records & assessment_records:
            raise ValueError('Recording leakage')
        validation = [r for r in rows if r['recording_id'] in validation_records]
        assessment = [r for r in rows if r['participant_id'] == fold['held_out_subject']]
        if any(r['participant_id'] == fold['held_out_subject'] for r in validation):
            raise ValueError('Person leakage')
        rule, audit = select_rule(validation, features, method, rules)
        outcomes = []
        for row in assessment:
            f = features[row['sample_id']][method]
            withheld = withhold_rate(f, rule)
            outcome = {**row, 'predictions': {**row['predictions'], 'current': row['predictions'][method],
                        'withheld': None if withheld else row['predictions'][method]},
                       'current_method': method, 'fold': name, 'features': f, 'rate_withheld': withheld,
                       'reason': rule['name'] if withheld else 'feature_unavailable' if not f['available'] else 'no_withholding',
                       'breathing_status_changed': False}
            outcomes.append(outcome)
            for control in rules:
                controls[control['name']].append({**outcome, 'rate_withheld': withhold_rate(f, control)})
        before, after = score_outcomes(outcomes), score_outcomes(outcomes, True)
        folds[name] = {'held_out_person': fold['held_out_subject'], 'method': method, 'rule': rule,
                       'inner_candidates': audit, 'validation_ids': [r['sample_id'] for r in validation],
                       'assessment_ids': [r['sample_id'] for r in assessment],
                       'before': before, 'after': after,
                       'outer_safety_pass': budget_pass(outcomes, 'current', [r['rate_withheld'] for r in outcomes])}
        pooled.extend(outcomes)
    if len(pooled) != len(rows) or len({r['sample_id'] for r in pooled}) != len(rows):
        raise ValueError('Outer assessments must cover each row exactly once')
    before, after = score_outcomes(pooled), score_outcomes(pooled, True)
    safety = budget_pass(pooled, 'current', [r['rate_withheld'] for r in pooled])
    benefit = after['large_interception_rate'] >= .25 and after['mae_bpm'] <= before['mae_bpm']
    type_counts = {}
    for row in pooled:
        kind = classify(row['predictions']['current'], row['reference_bpm'])
        if abs(row['predictions']['current'] - row['reference_bpm']) > 5:
            counts = type_counts.setdefault(kind, dict(total=0, withheld=0))
            counts['total'] += 1
            counts['withheld'] += int(row['rate_withheld'])
    report = {'config': config, 'before': before, 'after': after, 'folds': folds,
              'total_dataset_windows': len(dataset), 'total_window_coverage': after['retained']/len(dataset),
              'person_macro_mae_bpm_before': float(np.mean([f['before']['mae_bpm'] for f in folds.values()])),
              'person_macro_mae_bpm_after': float(np.mean([f['after']['mae_bpm'] for f in folds.values()])),
              'feature_unavailable': sum(not r['features']['available'] for r in pooled),
              'outer_safety_pass': safety, 'benefit_pass': benefit,
              'development_support_pass': safety and benefit, 'application_enabled': False,
              'large_error_types': type_counts,
              'by_environment': {env: {'before': score_outcomes([r for r in pooled if r['environment'] == env]),
                                       'after': score_outcomes([r for r in pooled if r['environment'] == env], True)} for env in ('Quiet','Interference')},
              'fixed_rule_diagnostics_not_for_posthoc_selection': {name: score_outcomes(outcomes, True) for name, outcomes in controls.items()},
              'withheld_reason_counts': dict(Counter(r['reason'] for r in pooled if r['rate_withheld']))}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2))
    (output / 'window_results.json').write_text(json.dumps(pooled, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ('config','folds','fixed_rule_diagnostics_not_for_posthoc_selection')}, indent=2))
    print('Selections:', {name: f['rule'] for name, f in folds.items()})


if __name__ == '__main__':
    main()
