"""Descriptive CSI candidate/reference audit; reference never chooses a prediction."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import find_peaks

from ..preprocessing.respiratory_rate import spectra, reference_rate
from ..preprocessing.wire_breath import WireBreathDataset
from .evaluate_spike_preprocessing import ROOT, candidate_window, digest

OUTPUT = ROOT / 'peak_evidence_v1'
TOLERANCE_BPM = 2.


def peak_evidence(x):
    """CSI-only candidates; support means dominant carrier peaks, not independence."""
    f, power = spectra(np.asarray(x)[0], band_hz=(.05, 1.))
    mean = power.mean(axis=0)
    allowed = (f >= .1) & (f <= .7)
    scale = max(float(mean[allowed].max()), 1e-12)
    carrier_bpm = f[allowed][np.argmax(power[:, allowed], axis=1)] * 60
    indices, properties = find_peaks(mean, prominence=0.)
    peaks = []
    for j, index in enumerate(indices):
        if not allowed[index]:
            continue
        bpm = float(f[index] * 60)
        peaks.append({'bpm': bpm, 'height': float(mean[index] / scale),
                      'prominence': float(properties['prominences'][j] / scale),
                      'carrier_support': float(np.mean(abs(carrier_bpm-bpm) <= TOLERANCE_BPM))})
    peaks.sort(key=lambda p: (-p['height'], p['bpm']))
    return {'peaks': peaks, 'carrier_dominant_bpm': carrier_bpm.tolist()}


def reference_evidence(signal):
    evidence = peak_evidence(np.asarray(signal)[None, None, :])
    peaks = evidence['peaks']
    ratio = peaks[1]['height']/peaks[0]['height'] if len(peaks) > 1 else 0.
    halves = [reference_rate(part) for part in np.array_split(signal, 2)]
    return {'peaks': peaks, 'second_first_height_ratio': ratio,
            'competing_peak_warning': ratio >= .5, 'half_window_bpm': halves,
            'temporal_discrepancy_warning': abs(halves[0]-halves[1]) > TOLERANCE_BPM}


def label_evidence(evidence, reference_bpm, predicted_bpm):
    near = [p for p in evidence['peaks'] if abs(p['bpm']-reference_bpm) <= TOLERANCE_BPM]
    closest = min(evidence['peaks'], key=lambda p: abs(p['bpm']-reference_bpm), default=None)
    return {'classification': 'reference_near_peak_present' if near else 'no_reference_near_mean_peak',
            'reference_near_peaks': near, 'closest_peak': closest,
            'half_frequency_relation': abs(2*predicted_bpm-reference_bpm) <= TOLERANCE_BPM,
            'double_frequency_relation': abs(predicted_bpm/2-reference_bpm) <= TOLERANCE_BPM}


def plot_case(row, sample, cleaned, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    for name, x, color in [('baseline', sample.x, '#365b89'), ('raw_hampel8', cleaned, '#d37b27')]:
        f, power = spectra(x[0], band_hz=(.05, 1.))
        curve = power.mean(axis=0)
        axes[0, 0].plot(f*60, curve/max(curve.max(), 1e-12), label=name, color=color)
        ev = row['evidence'][name]
        axes[0, 1].hist(ev['carrier_dominant_bpm'], bins=np.linspace(6, 42, 37),
                        histtype='step', label=name, color=color)
        for peak in ev['peaks'][:4]:
            axes[0, 0].annotate(f"{peak['bpm']:.1f}", (peak['bpm'], peak['height']), fontsize=7)
    axes[0, 0].axvline(row['reference_bpm'], color='black', linestyle=':', label='Reference (evaluation only)')
    axes[0, 0].set(xlim=(3, 60), xlabel='Cycles/min', ylabel='Relative mean power', title='CSI local candidates')
    axes[0, 1].set(xlabel='Carrier dominant cycles/min', ylabel='Carrier count', title='Correlated carriers: not independent votes')
    reference = sample.respiratory_reference
    axes[1, 0].plot(np.arange(len(reference))/30, reference, linewidth=.8)
    axes[1, 0].set(xlabel='Time (s)', ylabel='Reference amplitude', title='Reference waveform (evaluation only)')
    f, power = spectra(reference[None, :], band_hz=(.05, 1.))
    axes[1, 1].plot(f*60, power[0]/max(power[0].max(), 1e-12))
    axes[1, 1].axvline(row['reference_bpm'], color='black', linestyle=':')
    axes[1, 1].set(xlabel='Cycles/min', ylabel='Relative reference power', title='Reference spectrum; ambiguity flags are not exclusions')
    for ax in axes.flat:
        ax.grid(alpha=.2)
    for ax in axes[0]:
        ax.legend(fontsize=8)
    fig.suptitle(f"{row['sample_id']}\nReference {row['reference_bpm']:.2f}; original {row['predictions']['baseline']:.2f}; "
                 f"cleaned {row['predictions']['raw_hampel8']:.2f}; previous selected {row['previous_selected_bpm']:.2f}", fontsize=9)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    if OUTPUT.exists():
        raise FileExistsError('Preserve completed peak evidence audit')
    sources = ROOT/'spike_preprocessing_v1'
    previous = json.loads((sources/'selected_predictions.json').read_text())
    originals = json.loads((sources/'window_predictions.json').read_text())
    original_by_id = {r['sample_id']: r for r in originals}
    previous_by_id = {r['sample_id']: r for r in previous}
    targets = {r['sample_id'] for r in previous if abs(r['predictions']['selected']-r['reference_bpm']) > 5
               or (r['baseline_error'] <= 2 and abs(r['predictions']['selected']-r['reference_bpm']) > 2)}
    protected = json.loads((sources/'config.json').read_text())['protected_artifact_hashes']
    OUTPUT.mkdir()
    (OUTPUT/'config.json').write_text(json.dumps({'purpose': 'descriptive only, no rule selection',
        'protocol_document_sha256': digest(Path(__file__).parents[1]/'PEAK_EVIDENCE_PROTOCOL.md'),
        'source_predictions_sha256': digest(sources/'selected_predictions.json'),
        'audit_code_sha256': digest(__file__), 'tolerance_bpm': TOLERANCE_BPM,
        'protected_artifact_hashes': protected, 'application_enabled': False}, indent=2))
    dataset = WireBreathDataset('data/WiReBreath', window_seconds=30, stride_seconds=30, normalize=False)
    rows = []
    for i, ref in enumerate(dataset._windows):
        if ref.sample_id not in original_by_id:
            continue
        sample = dataset[i]
        assert sample.assessable
        source = original_by_id[ref.sample_id]
        raw = loadmat(ref.path, squeeze_me=True, variable_names=('time', 'csi_abs'))
        cleaned, _ = candidate_window(np.asarray(raw['time']).reshape(-1), raw['csi_abs'],
            ref.start_seconds+np.arange(900)/30, sample.x,
            {'placement': 'raw', 'threshold': 8.})
        evidence = {name: peak_evidence(x) for name, x in [('baseline', sample.x), ('raw_hampel8', cleaned)]}
        row = {k: source[k] for k in ('sample_id', 'recording_id', 'participant_id', 'reference_bpm', 'predictions', 'baseline_error')}
        row.update(evidence=evidence, reference=reference_evidence(sample.respiratory_reference),
                   previous_selected_bpm=previous_by_id[ref.sample_id]['predictions']['selected'],
                   target_case=ref.sample_id in targets)
        row['diagnosis'] = {name: label_evidence(ev, row['reference_bpm'], row['predictions'][name])
                            for name, ev in evidence.items()}
        if row['target_case']:
            path = OUTPUT/f"case_{sum(r['target_case'] for r in rows)+1:02d}.png"
            plot_case(row, sample, cleaned, path)
            row['plot_path'] = str(path)
        rows.append(row)
        if len(rows) % 50 == 0:
            print('Audited', len(rows), 'valid windows', flush=True)
    assert len(rows) == len(originals) and {r['sample_id'] for r in rows if r['target_case']} == targets
    cases = [r for r in rows if r['target_case']]
    summary = {'windows': len(rows), 'target_cases': len(cases),
        'previous_selected_large_errors': sum(abs(r['previous_selected_bpm']-r['reference_bpm']) > 5 for r in cases),
        'target_classification': {name: dict(Counter(r['diagnosis'][name]['classification'] for r in cases)) for name in evidence},
        'reference_competing_peak_warnings': sum(r['reference']['competing_peak_warning'] for r in cases),
        'reference_temporal_warnings': sum(r['reference']['temporal_discrepancy_warning'] for r in cases),
        'target_case_ids': [r['sample_id'] for r in cases], 'application_enabled': False}
    assert all(digest(path) == sha for path, sha in protected.items())
    (OUTPUT/'window_evidence.json').write_text(json.dumps(rows, indent=2))
    (OUTPUT/'audit_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
