"""Fixed common/relative representation pilot, never imported by live inference."""
import json
from pathlib import Path

import numpy as np

from ..preprocessing.common_mode import DECOMPOSITION_ID, decompose_common_mode
from ..preprocessing.respiratory_rate import spectra
from ..preprocessing.unified_rate import estimate_unified_rate
from ..preprocessing.wire_breath import WireBreathDataset
from ..preprocessing.wire_cache import load_wire_breath_cache
from ..preprocessing.wire_protocol import DEFAULT_PATH, load_protocol
from .audit_peak_evidence import peak_evidence, label_evidence
from .evaluate_local_rate_peaks import evaluate
from .evaluate_spike_preprocessing import ROOT, digest, select_preprocessing

METHODS = ['baseline', 'common', 'relative']


def component_result(x):
    try:
        estimate = estimate_unified_rate(x)
        evidence = peak_evidence(x)
        return {'available': True, 'estimate': estimate, 'evidence': evidence}
    except ValueError as error:
        return {'available': False, 'reason': str(error), 'estimate': None, 'evidence': None}


def describe_peaks(result, reference_bpm):
    if not result['available']:
        return {'available': False, 'reference_near_peak': False, 'best_near_rank': None,
                'best_near_height': None}
    evidence = result['evidence']
    diagnosis = label_evidence(evidence, reference_bpm, result['estimate']['bpm'])
    ranks = [i+1 for i, peak in enumerate(evidence['peaks']) if abs(peak['bpm']-reference_bpm) <= 2.]
    return {'available': True, 'reference_near_peak': bool(ranks),
            'best_near_rank': min(ranks) if ranks else None,
            'best_near_height': max((p['height'] for p in diagnosis['reference_near_peaks']), default=None)}


def plot_case(row, sample, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    common, relative, _ = decompose_common_mode(sample.x)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    t = np.arange(900)/30
    axes[0, 0].plot(t, common[0, 0], linewidth=.8, color='#d58129')
    axes[0, 0].set(title=f"Equal-weight common trace; energy {row['decomposition']['common_energy_fraction']:.1%}",
                   xlabel='Time (s)', ylabel='Standardized-input units')
    for index in (0, relative.shape[1]//2):
        axes[0, 1].plot(t, relative[0, index], linewidth=.7, alpha=.8, label=f'Valid carrier position {index}')
    axes[0, 1].set(title=f"Illustrative relative traces; total relative energy {row['decomposition']['relative_energy_fraction']:.1%}",
                   xlabel='Time (s)', ylabel='Standardized-input units')
    axes[0, 1].legend(fontsize=7)
    for name, x, color in [('baseline', sample.x, '#365b89'), ('common', common, '#d58129'), ('relative', relative, '#27836c')]:
        if not row['components'][name]['available']:
            continue
        f, power = spectra(x[0], band_hz=(.05, 1.))
        mean = power.mean(axis=0)
        allowed = (f >= .1) & (f <= .7)
        axes[1, 0].plot(f*60, mean/max(float(mean[allowed].max()), 1e-12), color=color,
                        label=f"{name}: {row['predictions'][name]:.2f} BPM")
    axes[1, 0].axvline(row['reference_bpm'], color='black', linestyle=':', label='Reference (evaluation only)')
    axes[1, 0].set(xlim=(3, 60), ylim=(0, 1.6), xlabel='Cycles/min', ylabel='Relative allowed-band power',
                   title='Separately normalized spectra, NOT physical energy')
    axes[1, 0].legend(fontsize=7)
    axes[1, 1].plot(t, sample.respiratory_reference, linewidth=.8)
    axes[1, 1].set(title=f"Unchanged respiratory reference: {row['reference_bpm']:.2f} BPM", xlabel='Time (s)', ylabel='Reference amplitude')
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle(row['sample_id']+'\nCommon/relative do not identify physical sources', fontsize=10)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    output = ROOT/'common_mode_v1'
    if output.exists():
        raise FileExistsError('Preserve common-mode experiment; no overwrite')
    sources = ROOT/'spike_preprocessing_v1'
    originals_path = sources/'window_predictions.json'
    originals = json.loads(originals_path.read_text())
    by_id = {r['sample_id']: r for r in originals}
    previous = json.loads((sources/'selected_predictions.json').read_text())
    harmed = {r['sample_id'] for r in previous if r['baseline_error'] <= 2 and abs(r['predictions']['selected']-r['reference_bpm']) > 2}
    target_ids = {r['sample_id'] for r in originals if r['baseline_error'] > 5} | harmed
    protected = json.loads((sources/'config.json').read_text())['protected_artifact_hashes']
    audits = json.loads((sources/'raw_audit.json').read_text())
    if len(by_id) != 394 or any(digest(path) != sha for path, sha in protected.items()):
        raise ValueError('Evaluation baseline/protected artifacts changed')
    if any(digest(a['source_path']) != a['raw_sha256'] for a in audits):
        raise ValueError('Raw files changed since source audit')
    cache = load_wire_breath_cache(ROOT/'wire_breath_cache', 1401)
    protocol = load_protocol(DEFAULT_PATH, cache.manifest['samples'])
    config = {'decomposition_id': DECOMPOSITION_ID, 'methods': METHODS, 'application_enabled': False,
        'protocol_document_sha256': digest(Path(__file__).parents[1]/'COMMON_MODE_PROTOCOL.md'),
        'implementation_sha256': digest(Path(__file__).parents[1]/'preprocessing/common_mode.py'),
        'evaluation_sha256': digest(__file__), 'baseline_source_sha256': digest(originals_path),
        'source_audit_sha256': digest(sources/'raw_audit.json'),
        'protocol_sample_fingerprint_sha256': protocol['sample_fingerprint_sha256'],
        'protected_artifact_hashes': protected, 'target_case_ids': sorted(target_ids),
        'selection': 'inner record-macro MAE with full coverage and zero original-good harm; ties fewer changes then baseline',
        'support': 'full coverage, zero good harm, >=25% fewer large errors, no person MAE worsening',
        'limitations': ['Retrospective observed development data, not independent final validation.',
            'Common/relative components are mathematical representations, not physical interference labels.',
            'Spectral restandardization can amplify tiny residuals; inspect pre-estimation energy fractions.',
            'No cleaning, reference-driven component selection or new mixing rule.']}
    output.mkdir()
    (output/'config.json').write_text(json.dumps(config, indent=2))
    dataset = WireBreathDataset('data/WiReBreath', window_seconds=30, stride_seconds=30, normalize=False)
    rows = []
    for i, ref in enumerate(dataset._windows):
        if ref.sample_id not in by_id:
            continue
        sample = dataset[i]
        original = by_id[sample.sample_id]
        if not sample.assessable:
            raise ValueError('Quality-passing set changed')
        common, relative, decomposition = decompose_common_mode(sample.x)
        components = {name: component_result(x) for name, x in [('baseline', sample.x), ('common', common), ('relative', relative)]}
        predictions = {name: result['estimate']['bpm'] if result['available'] else None for name, result in components.items()}
        if predictions['baseline'] != original['predictions']['baseline']:
            raise ValueError('Original estimator mismatch')
        row = {k: original[k] for k in ('sample_id', 'recording_id', 'participant_id', 'posture', 'room', 'environment', 'reference_bpm', 'baseline_error')}
        row.update(predictions=predictions, decomposition=decomposition, components=components,
                   target_case=sample.sample_id in target_ids,
                   previous_cleanup_harm=sample.sample_id in harmed)
        row['peak_diagnostics'] = {name: describe_peaks(result, row['reference_bpm']) for name, result in components.items()}
        if row['target_case']:
            path = output/f"case_{sum(r['target_case'] for r in rows)+1:02d}.png"
            plot_case(row, sample, path)
            row['plot_path'] = str(path)
        rows.append(row)
        if len(rows) % 50 == 0:
            print('Compared', len(rows), 'quality-passing windows', flush=True)
    if len(rows) != 394 or {r['sample_id'] for r in rows} != set(by_id):
        raise ValueError('Window coverage changed')
    folds, pooled = {}, []
    for name, fold in protocol['folds'].items():
        records = {part: {cache.manifest['samples'][j]['recording_id'] for j in fold[part]}
                   for part in ('train', 'validation', 'assessment')}
        if records['train'] & (records['validation'] | records['assessment']) or records['validation'] & records['assessment']:
            raise ValueError('Recording leakage')
        validation = [r for r in rows if r['recording_id'] in records['validation']]
        assessment = [r for r in rows if r['recording_id'] in records['assessment']]
        if any(r['participant_id'] == fold['held_out_subject'] for r in validation) or any(r['participant_id'] != fold['held_out_subject'] for r in assessment):
            raise ValueError('Person leakage')
        chosen, inner = select_preprocessing(validation, METHODS)
        outcomes = [{**r, 'predictions': {**r['predictions'], 'selected': r['predictions'][chosen]}, 'selected_method': chosen} for r in assessment]
        pooled.extend(outcomes)
        folds[name] = {'chosen': chosen, 'inner_candidates': inner,
            'validation_ids': [r['sample_id'] for r in validation], 'assessment_ids': [r['sample_id'] for r in assessment],
            'baseline': evaluate(outcomes, 'baseline'), 'selected': evaluate(outcomes, 'selected')}
    if len(pooled) != 394 or len({r['sample_id'] for r in pooled}) != 394:
        raise ValueError('Outer assessment coverage error')
    groups = {'all': rows, 'original_large': [r for r in rows if r['baseline_error'] > 5],
              'original_good': [r for r in rows if r['baseline_error'] <= 2]}
    fixed = {method: {'metrics': evaluate(rows, method),
        'by_person': {s: evaluate([r for r in rows if r['participant_id'] == s], method) for s in sorted({r['participant_id'] for r in rows})},
        'new_large_errors': sum(r['baseline_error'] <= 5 and r['predictions'][method] is not None and abs(r['predictions'][method]-r['reference_bpm']) > 5 for r in rows),
        'changed_rates': sum(r['predictions'][method] != r['predictions']['baseline'] for r in rows)} for method in METHODS}
    peak_summary = {group: {method: {'windows': len(members),
        'available': sum(r['peak_diagnostics'][method]['available'] for r in members),
        'reference_near_peak': sum(r['peak_diagnostics'][method]['reference_near_peak'] for r in members),
        'reference_near_top_rank': sum(r['peak_diagnostics'][method]['best_near_rank'] == 1 for r in members)}
        for method in METHODS} for group, members in groups.items()}
    baseline, selected = evaluate(pooled, 'baseline'), evaluate(pooled, 'selected')
    safety = selected['retained'] == 394 and selected['good_harm_rate'] == 0
    reduction = (baseline['large_errors']-selected['large_errors'])/baseline['large_errors']
    nonworse = all(f['selected']['mae_bpm'] is not None and f['selected']['mae_bpm'] <= f['baseline']['mae_bpm'] for f in folds.values())
    summary = {'fixed': fixed, 'baseline': baseline, 'selected': selected, 'folds': folds, 'peak_summary': peak_summary,
        'energy_fractions': {group: {'windows': len(members),
            'common_quantiles_0_25_50_75_100': np.quantile([r['decomposition']['common_energy_fraction'] for r in members], [0, .25, .5, .75, 1]).tolist()}
            for group, members in groups.items()},
        'max_energy_sum_error': max(r['decomposition']['energy_sum_error'] for r in rows),
        'max_reconstruction_error': max(r['decomposition']['reconstruction_max_error'] for r in rows),
        'outer_safety_pass': safety, 'large_error_reduction_fraction': reduction,
        'per_person_mae_nonworse': nonworse, 'development_support_pass': safety and reduction >= .25 and nonworse,
        'total_window_coverage': selected['retained']/467, 'application_enabled': False}
    if any(digest(path) != sha for path, sha in protected.items()) or any(digest(a['source_path']) != a['raw_sha256'] for a in audits):
        raise ValueError('Protected artifact or raw file changed')
    (output/'window_predictions.json').write_text(json.dumps(pooled, indent=2))
    (output/'metrics.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in ('folds', 'fixed')}, indent=2))
    print('Selections', {k: v['chosen'] for k, v in folds.items()})
    print('Fixed results', {k: v['metrics'] for k, v in fixed.items()})


if __name__ == '__main__':
    main()
