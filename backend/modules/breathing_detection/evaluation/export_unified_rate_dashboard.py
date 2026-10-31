"""Refresh rate values/metadata without changing dashboard layout or CSI examples."""
import json
from pathlib import Path
from backend.shared.export import write_replay
from ..preprocessing.unified_rate import ESTIMATOR_SPEC, ESTIMATOR_ID

ROOT = Path('backend/modules/breathing_detection/outputs')


def apply_unified_rate(payload):
    source = ROOT / 'respiratory_rate_unified_v1'
    report = json.loads((source / 'metrics.json').read_text())
    rows = {r['sample_id']: r for r in json.loads((source / 'window_predictions.json').read_text())}
    historical = {r['sample_id']: r for r in json.loads((ROOT / 'respiratory_rate_boundary_only_v1/window_predictions.json').read_text())}
    for record in payload['records']:
        for sample in record['samples']:
            row = rows.get(sample['id'])
            sample['historical_fold_selected_bpm'] = historical[sample['id']]['predictions']['selected'] if sample['id'] in historical else None
            sample['candidate_bpm'] = row['estimate']['bpm'] if row else None  # existing presentation contract
            sample['rate_details'] = row['estimate'] if row else None
            sample['rate_estimator_id'] = ESTIMATOR_ID
    payload['version'] = '0.3'
    payload['rate_estimator'] = {**ESTIMATOR_SPEC, 'evaluation': 'retrospective_development_not_independent_test'}
    payload['provenance']['rate_method'] = ESTIMATOR_ID + ' · fixed experimental development reference'
    payload['benchmarks'].update(rate=report['unified'], rate_person_mae=report['person_macro_mae_bpm'],
                                 rate_coverage=report['total_window_coverage'],
                                 rate_evaluation='retrospective_development_not_independent_test')
    for experiment in payload['experiments']:
        if experiment['name'] == 'Boundary-only rate correction':
            experiment['name'] = 'Historical fold-selected rate correction'
            experiment['note'] = 'Historical grouped experiment: 394 valid windows; 22 large errors. Not the current fixed estimator.'
        if experiment['name'] == 'Rate reliability screen':
            experiment['note'] = 'Historical fold-selected estimator; reliability gate failed safety and remains disabled.'
    name = 'Unified fixed rate estimator'
    if not any(e['name'] == name for e in payload['experiments']):
        payload['experiments'].append(dict(name=name, result='0.68 BPM retrospective MAE', decision='Experimental',
            note='Fixed boundary0 rule for everyone; 19 large errors. Previously observed development data, not independent validation.'))
    return payload


def main():
    payload = json.loads((ROOT / 'replay.json').read_text())
    write_replay('breathing_detection', apply_unified_rate(payload), bundle=True)
    print('Refreshed fixed unified rate samples and development metadata; other data preserved')


if __name__ == '__main__':
    main()
