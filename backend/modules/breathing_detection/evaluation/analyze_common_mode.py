"""Post-assessment harm inspection only; no additional candidate or thresholds."""
import json
from pathlib import Path

from ..preprocessing.wire_breath import WireBreathDataset
from .evaluate_common_mode import plot_case
from .evaluate_spike_preprocessing import ROOT, digest


def main():
    output = ROOT/'common_mode_v1'
    target = output/'harm_analysis.json'
    if target.exists():
        raise FileExistsError('Preserve descriptive common-mode harm analysis')
    rows = json.loads((output/'window_predictions.json').read_text())
    harmed = [r for r in rows if r['baseline_error'] <= 2 and r['predictions']['relative'] is not None
              and abs(r['predictions']['relative']-r['reference_bpm']) > 2]
    by_id = {r['sample_id']: r for r in harmed}
    dataset = WireBreathDataset('data/WiReBreath', window_seconds=30, stride_seconds=30, normalize=False)
    plots = []
    for i, ref in enumerate(dataset._windows):
        if ref.sample_id not in by_id:
            continue
        row = by_id[ref.sample_id]
        path = output/f'relative_harm_{len(plots)+1:02d}.png'
        if path.exists():
            raise FileExistsError('Preserve harm figure')
        plot_case(row, dataset[i], path)
        plots.append({'sample_id': row['sample_id'], 'path': str(path), 'predictions': row['predictions'],
                      'reference_bpm': row['reference_bpm'], 'decomposition': row['decomposition'],
                      'peak_diagnostics': row['peak_diagnostics']})
    assert len(plots) == len(harmed)
    result = {'purpose': 'post-assessment descriptive inspection; no algorithm/rule changes',
              'metrics_sha256': digest(output/'metrics.json'), 'predictions_sha256': digest(output/'window_predictions.json'),
              'analysis_code_sha256': digest(__file__), 'relative_harms': len(harmed),
              'common_accurate_among_relative_harms': sum(r['predictions']['common'] is not None and abs(r['predictions']['common']-r['reference_bpm']) <= 2 for r in harmed),
              'plots': plots, 'application_enabled': False}
    target.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'plots'}, indent=2))


if __name__ == '__main__':
    main()
