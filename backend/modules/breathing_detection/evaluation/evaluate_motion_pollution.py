"""Fixed grouped development assessment of CSI-only pollution withholding."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np

from ..preprocessing.motion_pollution import pollution_features, pollution_rules, calibrate_thresholds, pollution_withhold
from ..preprocessing.unified_rate import ESTIMATOR_ID
from ..preprocessing.wire_breath import WireBreathDataset
from ..preprocessing.wire_cache import load_wire_breath_cache
from ..preprocessing.wire_protocol import DEFAULT_PATH, load_protocol
from .evaluate_rate_reliability_stage2 import metrics, budget_pass

ROOT = Path('backend/modules/breathing_detection/outputs')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_pollution_rule(rows, features, thresholds, rules):
    selected, best_key, audit = None, (0,0,0), []
    for rule in rules:
        limits = thresholds[str(rule['quantile'])]
        mask = [pollution_withhold(features[r['sample_id']],rule,limits) for r in rows]
        m = metrics(rows,'unified',mask)
        eligible = budget_pass(rows,'unified',mask)
        audit.append({'rule': rule, 'eligible': eligible, 'metrics': m})
        key = (m['large_withheld'],-m['good_withheld'],-m['withheld'])
        if eligible and key > best_key:
            selected, best_key = rule, key
    return selected, audit


def score(rows, apply_withholding=False):
    return metrics(rows,'unified',[r['rate_withheld'] if apply_withholding else False for r in rows])


def assess(rows, features, rule, thresholds):
    outcomes = []
    for row in rows:
        f = features[row['sample_id']]
        mask = pollution_withhold(f,rule,thresholds[str(rule['quantile'])] if rule else {})
        outcomes.append({**row, 'features': f, 'rate_withheld': mask,
                         'predictions': {**row['predictions'],'withheld': None if mask else row['predictions']['unified']},
                         'reason': rule['name'] if mask else 'feature_unavailable' if not f['available'] else 'no_withholding',
                         'breathing_status_changed': False})
    return outcomes


def main():
    output = ROOT/'motion_pollution_v1'
    if output.exists():
        raise FileExistsError('Preserve configured pollution experiment; use a reviewed new version')
    source = ROOT/'respiratory_rate_unified_v1/window_predictions.json'
    rows = json.loads(source.read_text())
    if len(rows) != len({r['sample_id'] for r in rows}):
        raise ValueError('Duplicate source sample IDs')
    if any(r['estimate']['estimator_id'] != ESTIMATOR_ID or r['estimate']['bpm'] != r['predictions']['unified'] for r in rows):
        raise ValueError('Mixed/non-unified estimator source')
    cache = load_wire_breath_cache(ROOT/'wire_breath_cache',1401)
    protocol = load_protocol(DEFAULT_PATH,cache.manifest['samples'])
    metadata = {r['sample_id']:r for r in cache.manifest['samples'] if r['sample_id'].endswith('@0.000s')}
    for row in rows:
        if row['sample_id'] not in metadata or any(row[k] != metadata[row['sample_id']][k] for k in ('recording_id','participant_id','posture','room','environment')):
            raise ValueError('Source/cache metadata mismatch')
    protected = [source,ROOT/'respiratory_rate_unified_v1/metrics.json',ROOT/'lstm_interpolated_full_seed123/best_model.pt',
                 ROOT/'replay.json',Path('frontend/src/modules/breathing_detection/data/replay.json'),Path('frontend/package-lock.json')]
    rules = pollution_rules()
    config = {'estimator_id':ESTIMATOR_ID,'protocol_sha256':protocol['sample_fingerprint_sha256'],
              'source_predictions_sha256':digest(source),
              'feature_code_sha256':digest(Path(__file__).parents[1]/'preprocessing/motion_pollution.py'),
              'evaluation_code_sha256':digest(__file__),
              'metrics_code_sha256':digest(Path(__file__).with_name('evaluate_rate_reliability_stage2.py')),
              'rules':rules,'calibration':'record-balanced per-feature inverse empirical quantile of inner TRAIN CSI only',
              'selection':'inner validation only: maximize >5 BPM errors intercepted; ties fewer <=2 BPM accurate and total withheld; no gate fallback',
              'budgets':'pooled AND each person: <=1% accurate and <=15% total withheld in validation and assessment',
              'support':'outer safety AND >=25% large interception AND no worsening retained MAE; no automatic deployment',
              'ground_truth':'Interference/Quiet describe scenario, NOT window-level movement labels; reference only in rate-error selection/evaluation',
              'protected_artifact_hashes':{str(p):digest(p) for p in protected},
              'application_enabled':False,'limitations':['Already-observed development data; fixed estimator was retrospectively selected.',
                  'Not physical motion recognition or causal attribution; no labelled motion accuracy.',
                  'Five people, correlated windows/recordings and small discrete harm budgets.',
                  'Missing features preserve BPM for this offline experiment, not certify reliability.',
                  'Matched-error controls and fixed outer controls are descriptive only; no posthoc rule tuning.']}
    output.mkdir()
    (output/'config.json').write_text(json.dumps(config,indent=2))
    dataset = WireBreathDataset('data/WiReBreath',window_seconds=30,stride_seconds=30,normalize=False)
    by_id = {r['sample_id']:r for r in rows}
    features = {}
    for i in range(len(dataset)):
        sample = dataset[i]
        if sample.sample_id not in by_id:
            continue
        try:
            features[sample.sample_id] = pollution_features(sample.x)
        except ValueError as error:
            features[sample.sample_id] = dict(available=False,reason=str(error))
        if len(features)%100 == 0:
            print('Extracted',len(features),'CSI-only pollution feature sets',flush=True)
    if set(features) != set(by_id):
        raise ValueError('Incomplete feature coverage')
    (output/'features.json').write_text(json.dumps(features,indent=2))
    folds, pooled, controls = {}, [], {r['name']:[] for r in rules}
    for name,fold in protocol['folds'].items():
        records = {part:{cache.manifest['samples'][i]['recording_id'] for i in fold[part]} for part in ('train','validation','assessment')}
        if any(records[a]&records[b] for a,b in [('train','validation'),('train','assessment'),('validation','assessment')]):
            raise ValueError('Recording leakage')
        train = [r for r in rows if r['recording_id'] in records['train']]
        validation = [r for r in rows if r['recording_id'] in records['validation']]
        assessment = [r for r in rows if r['recording_id'] in records['assessment']]
        if any(r['participant_id']==fold['held_out_subject'] for r in train+validation) or any(r['participant_id']!=fold['held_out_subject'] for r in assessment):
            raise ValueError('Person leakage')
        limits = {str(q):calibrate_thresholds(train,features,q) for q in (.9,.95)}
        rule,audit = select_pollution_rule(validation,features,limits,rules)
        outcomes = assess(assessment,features,rule,limits)
        pooled.extend(outcomes)
        folds[name] = {'held_out_person':fold['held_out_subject'],'selected_rule':rule,
                       'train_ids':[r['sample_id'] for r in train],'validation_ids':[r['sample_id'] for r in validation],
                       'assessment_ids':[r['sample_id'] for r in assessment],'thresholds':limits,'inner_candidates':audit,
                       'before':score(outcomes),'after':score(outcomes,True),
                       'outer_safety_pass':budget_pass(outcomes,'unified',[r['rate_withheld'] for r in outcomes])}
        for control in rules:
            controls[control['name']].extend(assess(assessment,features,control,limits))
    if len(pooled)!=len(rows) or len({r['sample_id'] for r in pooled})!=len(rows):
        raise ValueError('Outer coverage is not exactly one assessment per source sample')
    before,after = score(pooled),score(pooled,True)
    safety = budget_pass(pooled,'unified',[r['rate_withheld'] for r in pooled])
    benefit = after['large_interception_rate']>=.25 and after['mae_bpm']<=before['mae_bpm']
    report = {'config':config,'before':before,'after':after,'folds':folds,
              'total_segments':len(dataset),'total_window_coverage':after['retained']/len(dataset),
              'person_macro_mae_before':float(np.mean([f['before']['mae_bpm'] for f in folds.values()])),
              'person_macro_mae_after':float(np.mean([f['after']['mae_bpm'] for f in folds.values()])),
              'feature_unavailable':sum(not f['available'] for f in features.values()),
              'outer_safety_pass':safety,'benefit_pass':benefit,'development_support_pass':safety and benefit,
              'application_enabled':False,'reason_counts':dict(Counter(r['reason'] for r in pooled if r['rate_withheld'])),
              'by_environment':{env:{'before':score([r for r in pooled if r['environment']==env]),
                                    'after':score([r for r in pooled if r['environment']==env],True)} for env in ('Quiet','Interference')},
              'fixed_controls_not_for_posthoc_selection':{name:{'metrics':score(outcomes,True),
                    'outer_safety_pass':budget_pass(outcomes,'unified',[r['rate_withheld'] for r in outcomes])}
                    for name,outcomes in controls.items()}}
    if any(digest(path)!=expected for path,expected in config['protected_artifact_hashes'].items()):
        raise ValueError('A protected model/rate/replay artifact changed during experiment')
    (output/'window_results.json').write_text(json.dumps(pooled,indent=2))
    (output/'metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('config','folds','fixed_controls_not_for_posthoc_selection','by_environment')},indent=2))
    print('Selections:',{name:f['selected_rule'] for name,f in folds.items()})


if __name__ == '__main__':
    main()
