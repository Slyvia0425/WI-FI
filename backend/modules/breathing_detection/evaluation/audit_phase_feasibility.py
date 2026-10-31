"""Complex-CSI phase source audit and fixed exploratory spectral readouts."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from ..preprocessing.phase_diagnostics import PHASE_METHODS, phase_source_audit, phase_windows
from ..preprocessing.respiratory_rate import spectra, reference_rate
from ..preprocessing.wire_breath import WireBreathDataset
from ..preprocessing.wire_cache import load_wire_breath_cache
from ..preprocessing.wire_protocol import DEFAULT_PATH, load_protocol
from .evaluate_common_mode import component_result, describe_peaks
from .evaluate_local_rate_peaks import evaluate
from .evaluate_spike_preprocessing import ROOT, digest, select_preprocessing

METHODS = ['baseline', *PHASE_METHODS]


def plot_case(row, sample, time, h, targets, windows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), layout='constrained')
    stats = row['phase_window_audit']
    lo = max(0, int(np.searchsorted(time, targets[0], side='right'))-1)
    hi = min(len(time), int(np.searchsorted(time, targets[-1], side='left'))+1)
    t, source = time[lo:hi]-targets[0], np.asarray(h[lo:hi], dtype=np.complex128)
    indices = stats['valid_column_indices']
    pairs = stats['valid_pair_left_indices']
    if indices:
        phase = np.angle(source[:, indices[0]])
        axes[0, 0].plot(t, phase, linewidth=.6)
        increment = np.angle(source[1:, indices[0]]*np.conj(source[:-1, indices[0]]))
        axes[0, 1].plot(t[1:], increment, linewidth=.6, label='First valid carrier increment')
    if pairs:
        difference = np.angle(source[:, pairs[0]+1]*np.conj(source[:, pairs[0]]))
        axes[0, 1].plot(t[1:], np.angle(np.exp(1j*np.diff(difference))), linewidth=.6, alpha=.8, label='First valid pair increment')
    axes[0, 0].set(title='First valid carrier, wrapped source phase', xlabel='Native time (s)', ylabel='Radians')
    axes[0, 1].set(title='Circular temporal increments (not hardware labels)', xlabel='Native time (s)', ylabel='Radians')
    axes[0, 1].legend(fontsize=7)
    colors = {'baseline': '#555555', 'phase_raw': '#365b89', 'phase_adjacent': '#27836c', 'phase_index_residual': '#d58129'}
    for method, x in windows.items():
        trace = x[0, 0].astype(float)
        axes[1, 0].plot(np.arange(900)/30, (trace-trace.mean())/max(float(trace.std()), 1e-8),
                        linewidth=.7, color=colors[method], label=method)
    axes[1, 0].set(title='First valid trace, separately standardized for display', xlabel='Time (s)', ylabel='Display-standardized phase')
    axes[1, 0].legend(fontsize=7)
    for method, x in {'baseline': sample.x, **windows}.items():
        if row['predictions'][method] is None:
            continue
        f, power = spectra(x[0], band_hz=(.05, 1.))
        curve = power.mean(axis=0)
        allowed = (f >= .1) & (f <= .7)
        axes[1, 1].plot(f*60, curve/max(float(curve[allowed].max()), 1e-12),
                        color=colors[method], label=f"{method}: {row['predictions'][method]:.2f} BPM")
    axes[1, 1].axvline(row['reference_bpm'], color='black', linestyle=':', label='Reference, evaluation only')
    axes[1, 1].set(title='Separately normalized mean spectra', xlabel='Cycles/min', ylabel='Relative allowed-band power', xlim=(3, 60), ylim=(0, 1.6))
    axes[1, 1].legend(fontsize=6)
    axes[2, 0].plot(np.arange(900)/30, sample.respiratory_reference, linewidth=.8)
    axes[2, 0].set(title=f"Unchanged reference: {row['reference_bpm']:.2f} BPM", xlabel='Time (s)', ylabel='Reference amplitude')
    if indices:
        axes[2, 1].imshow(np.angle(source[:, indices[:32]]).T, aspect='auto', origin='lower',
                          extent=(float(t[0]), float(t[-1]), 0, min(len(indices), 32)), vmin=-np.pi, vmax=np.pi, cmap='twilight')
    axes[2, 1].set(title='First 32 valid columns: wrapped phase (not frequency calibration)', xlabel='Native time (s)', ylabel='Valid column position')
    for ax in axes.flat:
        ax.grid(alpha=.15)
    fig.suptitle(f"{row['sample_id']}\n{stats['valid_carriers']} valid carriers / {stats['valid_adjacent_pairs']} adjacent pairs; index projection is NOT certified calibration", fontsize=9)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    output = ROOT/'phase_feasibility_v1'
    if output.exists():
        raise FileExistsError('Preserve phase feasibility experiment')
    prior = ROOT/'spike_preprocessing_v1'
    originals_path = prior/'window_predictions.json'
    originals = json.loads(originals_path.read_text())
    by_id = {r['sample_id']: r for r in originals}
    raw_sources = json.loads((prior/'raw_audit.json').read_text())
    protected = json.loads((prior/'config.json').read_text())['protected_artifact_hashes']
    if any(digest(path) != sha for path, sha in protected.items()) or any(digest(a['source_path']) != a['raw_sha256'] for a in raw_sources):
        raise ValueError('Protected application artifact/raw source changed')
    cache = load_wire_breath_cache(ROOT/'wire_breath_cache', 1401)
    protocol = load_protocol(DEFAULT_PATH, cache.manifest['samples'])
    targets_to_plot = {r['sample_id'] for r in sorted([r for r in originals if r['baseline_error'] > 5], key=lambda r: (-r['baseline_error'], r['sample_id']))[:3]}
    targets_to_plot.add('Room2_50Hz_Subject04_Freefaller_Interference_Session0817_Trial01_seg002@0.000s')
    config = {'methods': METHODS, 'application_enabled': False, 'purpose': 'phase feasibility, not certified calibration',
        'protocol_document_sha256': digest(Path(__file__).parents[1]/'PHASE_FEASIBILITY_PROTOCOL.md'),
        'implementation_sha256': digest(Path(__file__).parents[1]/'preprocessing/phase_diagnostics.py'),
        'evaluation_sha256': digest(__file__), 'baseline_source_sha256': digest(originals_path),
        'raw_audit_sha256': digest(prior/'raw_audit.json'),
        'protocol_sample_fingerprint_sha256': protocol['sample_fingerprint_sha256'],
        'protected_artifact_hashes': protected, 'plot_ids': sorted(targets_to_plot),
        'zero_amplitude_tolerance': 1e-12, 'selection': 'inner record-macro MAE; full coverage and zero original-good harm; ties fewer changes then baseline',
        'support': 'full coverage, zero harm, >=25% fewer large errors, no person MAE worsening',
        'limitations': ['Previously observed overlapping development segments, not independent final validation.',
            'No verified antenna identities, physical carrier coordinates or calibration signals are assumed.',
            'Unwrap and index-affine projection are diagnostic assumptions, not hardware phase-error recovery.',
            'No training, fusion, reference-driven carrier/representation choice or quality upgrade.']}
    output.mkdir()
    (output/'config.json').write_text(json.dumps(config, indent=2))
    dataset = WireBreathDataset('data/WiReBreath', window_seconds=30, stride_seconds=30, normalize=False)
    rows, audits = [], []
    for i, ref in enumerate(dataset._windows):
        raw = loadmat(ref.path, squeeze_me=True, struct_as_record=False, variable_names=('time', 'csi_complex', 'metadata'))
        h = np.asarray(raw['csi_complex'])
        time = np.asarray(raw['time'], dtype=float).reshape(-1)
        targets = ref.start_seconds+np.arange(900)/30
        source_stats = phase_source_audit(h)
        windows, failures, window_stats = phase_windows(time, h, targets)
        audits.append({'sample_id': ref.sample_id, 'recording_id': ref.recording_id,
                       'source': source_stats, 'window': window_stats,
                       'metadata_fields': list(raw['metadata']._fieldnames),
                       'assessable': ref.sample_id in by_id, 'representation_failures': failures})
        if ref.sample_id in by_id:
            sample = dataset[i]
            original = by_id[ref.sample_id]
            if not sample.assessable or abs(reference_rate(sample.respiratory_reference)-original['reference_bpm']) > 1e-6:
                raise ValueError('Quality/reference changed')
            components = {'baseline': component_result(sample.x)}
            for method in PHASE_METHODS:
                components[method] = component_result(windows[method]) if method in windows else {
                    'available': False, 'estimate': None, 'evidence': None, 'reason': failures[method]}
            predictions = {method: result['estimate']['bpm'] if result['available'] else None for method, result in components.items()}
            if predictions['baseline'] != original['predictions']['baseline']:
                raise ValueError('Frozen amplitude estimator mismatch')
            row = {k: original[k] for k in ('sample_id', 'recording_id', 'participant_id', 'room', 'posture', 'environment', 'reference_bpm', 'baseline_error')}
            row.update(predictions=predictions, components=components, phase_window_audit=window_stats,
                       peak_diagnostics={method: describe_peaks(result, original['reference_bpm']) for method, result in components.items()})
            if ref.sample_id in targets_to_plot:
                path = output/f'case_{sum("plot_path" in r for r in rows)+1:02d}.png'
                plot_case(row, sample, time, h, targets, windows, path)
                row['plot_path'] = str(path)
            rows.append(row)
        if (i+1) % 50 == 0:
            print('Audited', i+1, 'source files; scored', len(rows), 'quality-passing windows', flush=True)
    if len(audits) != 467 or len(rows) != 394 or set(by_id) != {r['sample_id'] for r in rows}:
        raise ValueError('Phase audit/source coverage mismatch')
    folds, pooled = {}, []
    for name, fold in protocol['folds'].items():
        records = {part: {cache.manifest['samples'][j]['recording_id'] for j in fold[part]} for part in ('train', 'validation', 'assessment')}
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
    if len(pooled) != len({r['sample_id'] for r in pooled}) or len(pooled) != 394:
        raise ValueError('Duplicate/missing outer phase assessment')
    fixed = {method: {'metrics': evaluate(rows, method),
        'by_person': {s: evaluate([r for r in rows if r['participant_id'] == s], method) for s in sorted({r['participant_id'] for r in rows})},
        'unavailable_reasons': dict(Counter(r['components'][method]['reason'] for r in rows if not r['components'][method]['available'])),
        'new_large_errors': sum(r['baseline_error'] <= 5 and r['predictions'][method] is not None and abs(r['predictions'][method]-r['reference_bpm']) > 5 for r in rows),
        'original_large_reference_near_peak': sum(r['baseline_error'] > 5 and r['peak_diagnostics'][method]['reference_near_peak'] for r in rows)} for method in METHODS}
    baseline, selected = evaluate(pooled, 'baseline'), evaluate(pooled, 'selected')
    safety = selected['retained'] == 394 and selected['good_harm_rate'] == 0
    reduction = (baseline['large_errors']-selected['large_errors'])/baseline['large_errors']
    nonworse = all(f['selected']['mae_bpm'] is not None and f['selected']['mae_bpm'] <= f['baseline']['mae_bpm'] for f in folds.values())
    source_summary = {'files': len(audits), 'all_complex': True,
        'shapes': dict(Counter(str(a['source']['shape']) for a in audits)),
        'dtypes': dict(Counter(a['source']['dtype'] for a in audits)),
        'all_finite': all(a['source']['all_finite'] for a in audits),
        'metadata_fields': sorted({key for a in audits for key in a['metadata_fields']}),
        'files_with_zero_coefficients': sum(a['source']['zero_coefficients'] > 0 for a in audits),
        'source_valid_carrier_counts': dict(Counter(a['source']['valid_carriers'] for a in audits)),
        'source_wholly_zero_carrier_counts': dict(Counter(a['source']['wholly_zero_carriers'] for a in audits)),
        'source_intermittent_invalid_carrier_counts': dict(Counter(a['source']['intermittent_invalid_carriers'] for a in audits)),
        'valid_window_carrier_counts': dict(Counter(r['phase_window_audit']['valid_carriers'] for r in rows)),
        'valid_window_adjacent_pair_counts': dict(Counter(r['phase_window_audit']['valid_adjacent_pairs'] for r in rows))}
    for key in ('temporal_wrapped_increment_abs_median', 'temporal_wrapped_increment_gt_pi_over2_fraction',
                'temporal_increment_coherence_median', 'adjacent_difference_coherence_median'):
        values = [a['window'][key] for a in audits if key in a['window']]
        source_summary[key+'_quantiles_0_50_100'] = np.quantile(values, [0, .5, 1]).tolist() if values else None
    report = {'source_summary': source_summary, 'fixed': fixed, 'folds': folds, 'baseline': baseline, 'selected': selected,
        'outer_safety_pass': safety, 'large_error_reduction_fraction': reduction, 'per_person_mae_nonworse': nonworse,
        'development_support_pass': safety and reduction >= .25 and nonworse,
        'total_window_coverage': selected['retained']/467, 'application_enabled': False}
    if any(digest(path) != sha for path, sha in protected.items()) or any(digest(a['source_path']) != a['raw_sha256'] for a in raw_sources):
        raise ValueError('Protected application artifact/raw data changed during audit')
    (output/'source_audit.json').write_text(json.dumps(audits, indent=2))
    (output/'window_predictions.json').write_text(json.dumps(pooled, indent=2))
    (output/'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ('fixed', 'folds')}, indent=2))
    print('Fixed results', {method: values['metrics'] for method, values in fixed.items()})
    print('Selections', {name: f['chosen'] for name, f in folds.items()})


if __name__ == '__main__':
    main()
