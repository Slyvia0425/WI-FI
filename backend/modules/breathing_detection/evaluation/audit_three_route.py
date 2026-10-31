"""Audit three-route separation without fitting a prediction rule."""
import json
from pathlib import Path

import numpy as np

from ..preprocessing.three_route import three_route_features
from .evaluate_spike_preprocessing import ROOT, digest


def group_summary(rows):
    result = {'windows': len(rows), 'proposals_gt2_bpm': sum(r['features']['proposes_change'] for r in rows)}
    for key in ('baseline_common_agree', 'common_relative_agree', 'all_available'):
        result[key] = sum(r['features'][key] for r in rows)
    for route in ('baseline', 'common', 'relative'):
        for target in ('baseline', 'relative'):
            matches = [r['features']['matched_peaks'][route][target] for r in rows]
            found = [p for p in matches if p is not None]
            result[f'{route}_supports_{target}'] = {'matched': len(found),
                'height_min_median_max': np.quantile([p['height'] for p in found], [0, .5, 1]).tolist() if found else None,
                'prominence_min_median_max': np.quantile([p['prominence'] for p in found], [0, .5, 1]).tolist() if found else None}
    return result


def main():
    output = ROOT/'three_route_audit_v1'
    if output.exists():
        raise FileExistsError('Preserve three-route audit')
    source = ROOT/'common_mode_v1/window_predictions.json'
    protected = json.loads((ROOT/'common_mode_v1/config.json').read_text())['protected_artifact_hashes']
    if any(digest(path) != sha for path, sha in protected.items()):
        raise ValueError('Protected application artifact changed')
    output.mkdir()
    config = {'source_sha256': digest(source), 'audit_code_sha256': digest(__file__),
        'feature_code_sha256': digest(Path(__file__).parents[1]/'preprocessing/three_route.py'),
        'protocol_document_sha256': digest(Path(__file__).parents[1]/'THREE_ROUTE_PROTOCOL.md'),
        'protected_artifact_hashes': protected, 'application_enabled': False,
        'purpose': 'retrospective descriptive comparison, no new rule or thresholds'}
    (output/'config.json').write_text(json.dumps(config, indent=2))
    sources = json.loads(source.read_text())
    rows = []
    for row in sources:
        # Deliberately pass CSI components only, excluding reference diagnostics.
        features = three_route_features(row['components'])
        relative = row['predictions']['relative']
        relative_error = abs(relative-row['reference_bpm']) if relative is not None else None
        corrected = row['baseline_error'] > 5 and relative_error is not None and relative_error <= 2
        harmed = row['baseline_error'] <= 2 and (relative_error is None or relative_error > 2)
        rows.append({key: row[key] for key in ('sample_id', 'recording_id', 'participant_id', 'reference_bpm', 'baseline_error')} |
                    {'features': features, 'relative_error': relative_error,
                     'corrected_large': corrected, 'harmed_good': harmed})
    assert len(rows) == len({r['sample_id'] for r in rows}) == 394
    groups = {'all': rows, 'proposed_changes': [r for r in rows if r['features']['proposes_change']],
              'corrected_large': [r for r in rows if r['corrected_large']],
              'harmed_good': [r for r in rows if r['harmed_good']]}
    summary = {'groups': {name: group_summary(members) for name, members in groups.items()},
               'application_enabled': False, 'no_rule_evaluated': True}
    if digest(source) != config['source_sha256'] or any(digest(path) != sha for path, sha in protected.items()):
        raise ValueError('Source/protected artifact changed')
    (output/'features.json').write_text(json.dumps(rows, indent=2))
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    for group in ('corrected_large', 'harmed_good'):
        print(group)
        for r in groups[group]:
            f = r['features']
            print(r['sample_id'], 'rates', f['rates'], 'original-common', f['baseline_common_agree'], 'common-relative', f['common_relative_agree'],
                  'original proposal peak', f['matched_peaks']['baseline']['relative'],
                  'common original peak', f['matched_peaks']['common']['baseline'],
                  'common proposal peak', f['matched_peaks']['common']['relative'])


if __name__ == '__main__':
    main()
