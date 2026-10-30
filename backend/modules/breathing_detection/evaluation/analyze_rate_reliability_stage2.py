"""Post-run descriptive diagnostics only: never fit or modify stage-2 rules."""
import json
from pathlib import Path
import numpy as np
from .analyze_rate_errors import classify

ROOT = Path('backend/modules/breathing_detection/outputs/respiratory_rate_reliability_stage2_v1')


def describe(rows):
    valid = [r for r in rows if r['features']['available']]
    return {'windows': len(rows), 'feature_available': len(valid),
            'zero_disagreements': sum(r['features']['temporal_disagreement_count'] == 0 for r in valid),
            'at_least2_disagreements': sum(r['features']['temporal_disagreement_count'] >= 2 for r in valid),
            'all3_disagree': sum(r['features']['temporal_disagreement_count'] == 3 for r in valid),
            'harmonic_ratio_atleast_half': sum(r['features']['harmonic_competition_ratio'] >= .5 for r in valid),
            'median_spread_bpm': float(np.median([r['features']['subwindow_spread_bpm'] for r in valid])) if valid else None,
            'median_harmonic_ratio': float(np.median([r['features']['harmonic_competition_ratio'] for r in valid])) if valid else None}


def main():
    target = ROOT / 'diagnostics.json'
    if target.exists():
        raise FileExistsError('Preserve descriptive diagnostics')
    rows = json.loads((ROOT / 'window_results.json').read_text())
    for row in rows:
        row['current_error_bpm'] = abs(row['predictions']['current'] - row['reference_bpm'])
    large = sorted([r for r in rows if r['current_error_bpm'] > 5], key=lambda r: -r['current_error_bpm'])
    report = {'purpose': 'post-run descriptive only; no threshold or rule selection',
              'by_error': {'accurate_le2': describe([r for r in rows if r['current_error_bpm'] <= 2]),
                           'intermediate_2to5': describe([r for r in rows if 2 < r['current_error_bpm'] <= 5]),
                           'large_gt5': describe(large)},
              'large_by_type': {kind: describe([r for r in large if classify(r['predictions']['current'], r['reference_bpm']) == kind])
                               for kind in sorted({classify(r['predictions']['current'], r['reference_bpm']) for r in large})},
              'large_error_cases': [dict(sample_id=r['sample_id'], participant_id=r['participant_id'], recording_id=r['recording_id'],
                                        environment=r['environment'], candidate_bpm=r['predictions']['current'],
                                        reference_bpm=r['reference_bpm'], error_bpm=r['current_error_bpm'],
                                        error_type=classify(r['predictions']['current'], r['reference_bpm']), features=r['features']) for r in large]}
    target.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'large_error_cases'}, indent=2))
    print('Worst 3:', json.dumps(report['large_error_cases'][:3], indent=2))


if __name__ == '__main__':
    main()
