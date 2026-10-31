"""Paired waveform/spectrum review of all 35 large respiratory-rate errors."""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.signal import butter, detrend, find_peaks, sosfiltfilt

from backend.modules.breathing_detection.preprocessing.respiratory_rate import reference_rate, spectra, estimate_rates
from backend.modules.breathing_detection.preprocessing.wire_breath import WireBreathDataset


def waveform_evidence(sample, estimate, reference):
    signal = detrend(sample.respiratory_reference.astype(float))
    filtered = sosfiltfilt(butter(3, [.08, .8], btype='bandpass', fs=30, output='sos'), signal)
    filtered /= max(float(filtered.std()), 1e-8)
    peaks, _ = find_peaks(filtered, distance=int(30 / .7), prominence=.5)
    intervals = np.diff(peaks) / 30
    peak_rate = float(60 / np.median(intervals)) if len(intervals) >= 3 else None
    interval_cv = float(intervals.std() / intervals.mean()) if len(intervals) else None
    f, reference_power = spectra(sample.respiratory_reference[None])
    power = reference_power[0]
    csi_f, csi_power = spectra(sample.x[0])
    mean = csi_power.mean(axis=0)
    tolerance = 1 / 30

    def relative_peak(values, target_bpm):
        selected = abs(f - target_bpm / 60) <= tolerance
        return float(values[selected].max() / max(float(values.max()), 1e-12)) if selected.any() else None

    ref_half = relative_peak(power, reference / 2)
    ref_double = relative_peak(power, reference * 2)
    half_disagreement = peak_rate is not None and abs(peak_rate - reference / 2) <= 2 and interval_cv <= .25
    double_disagreement = peak_rate is not None and abs(peak_rate - 2 * reference) <= 2 and interval_cv <= .25
    spectral_secondary = ((ref_half is not None and ref_half >= .35) or
                          (ref_double is not None and ref_double >= .35))
    # Evidence tags are overlapping diagnostics, not causal truth labels.
    tags = []
    if half_disagreement or double_disagreement:
        tags.append('reference_peak_count_disagrees_by_factor_two')
    if spectral_secondary:
        tags.append('reference_secondary_harmonic_energy')
    ref_parts = [reference_rate(sample.respiratory_reference[start:start + 300]) for start in (0, 300, 600)]
    csi_parts = [estimate_rates(sample.x[..., start:start + 300])['mean_spectrum'] for start in (0, 300, 600)]
    if max(ref_parts) - min(ref_parts) > 6:
        tags.append('reference_short_window_frequency_varies')
    matching = relative_peak(mean, reference)
    if matching is not None and matching >= .5 and abs(estimate - reference) > 5:
        tags.append('csi_retains_energy_near_reference')
    elif matching is not None and matching < .2:
        tags.append('weak_csi_energy_near_reference')
    if peak_rate is not None and interval_cv <= .25 and abs(peak_rate - reference) <= 2:
        tags.append('reference_peak_count_supports_spectral_rate')
    if abs(estimate - f[0] * 60) < 1e-6:
        tags.append('selected_exact_lower_search_boundary')
    if abs(estimate - f[-1] * 60) < 1e-6:
        tags.append('selected_exact_upper_search_boundary')
    return {
        'reference_peak_count_bpm': peak_rate, 'reference_interval_cv': interval_cv,
        'reference_detected_peaks': len(peaks),
        'reference_half_relative_power': ref_half, 'reference_double_relative_power': ref_double,
        'csi_relative_power_near_reference': matching,
        'reference_10s_bpm': ref_parts, 'csi_10s_bpm': csi_parts, 'tags': tags,
    }, (filtered, peaks, f, power, mean, csi_power)


def select_control(case, good):
    # Prefer same source recording, then person/posture/environment. No nearest-spectrum fitting.
    def key(row):
        tier = (0 if row['recording_id'] == case['recording_id'] else
                1 if all(row[k] == case[k] for k in ('participant_id', 'posture', 'environment')) else
                2 if all(row[k] == case[k] for k in ('participant_id', 'posture')) else
                3 if row['participant_id'] == case['participant_id'] else 4)
        return tier, row['sample_id']
    control = min(good, key=key)
    return control, key(control)[0]


def plot_pair(case, control, case_sample, control_sample, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    for column, (row, sample) in enumerate(((case, case_sample), (control, control_sample))):
        estimate = row['estimates']['mean_spectrum']
        reference = row['reference_bpm']
        evidence, data = waveform_evidence(sample, estimate, reference)
        filtered, peaks, f, ref_power, mean, csi_power = data
        t = np.arange(900) / 30
        raw = detrend(sample.respiratory_reference.astype(float))
        raw /= max(raw.std(), 1e-8)
        axes[0, column].plot(t, raw, label='reference detrended', alpha=.5)
        axes[0, column].plot(t, filtered, label='reference 0.08-0.8 Hz')
        axes[0, column].plot(t[peaks], filtered[peaks], 'o', markersize=3)
        axes[0, column].set_title(('Error' if column == 0 else 'Accurate control') + f': CSI {estimate:.2f}, reference {reference:.2f} BPM\n' + sample.sample_id.replace('@', '\n@'), fontsize=9)
        axes[0, column].legend(fontsize=7)
        ref_index = np.argmin(abs(f - reference / 60))
        estimate_index = np.argmin(abs(f - estimate / 60))
        for label, carrier in [('reference-supporting carrier', int(np.argmax(csi_power[:, ref_index]))),
                               ('estimate-supporting carrier', int(np.argmax(csi_power[:, estimate_index])))]:
            values = detrend(sample.x[0, carrier].astype(float))
            values = sosfiltfilt(butter(3, [.08, .8], btype='bandpass', fs=30, output='sos'), values)
            values /= max(values.std(), 1e-8)
            axes[1, column].plot(t, values, label=f'{label} #{carrier}', alpha=.8)
        axes[1, column].legend(fontsize=7)
        axes[2, column].plot(f * 60, ref_power / ref_power.max(), label='reference spectrum')
        axes[2, column].plot(f * 60, mean / mean.max(), label='mean CSI spectrum')
        axes[2, column].axvline(reference, color='green', linestyle='--')
        axes[2, column].axvline(estimate, color='red', linestyle=':')
        axes[2, column].set_xlabel('BPM'); axes[2, column].legend(fontsize=7)
        axes[3, column].plot([5, 15, 25], evidence['reference_10s_bpm'], 'o-', label='reference 10s')
        axes[3, column].plot([5, 15, 25], evidence['csi_10s_bpm'], 'o-', label='CSI 10s')
        axes[3, column].set_xlabel('Window midpoint (s)'); axes[3, column].set_ylabel('BPM')
        axes[3, column].legend(fontsize=7)
    fig.suptitle('Diagnostic review: selected CSI carriers use reference information for illustration only', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(output, dpi=110)
    plt.close(fig)


def main():
    base = Path('backend/modules/breathing_detection/outputs/respiratory_rate_baseline_v1')
    output = base / 'waveform_review'
    output.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/breathing-matplotlib')
    os.environ.setdefault('XDG_CACHE_HOME', '/private/tmp/breathing-font-cache')
    import matplotlib
    matplotlib.use('Agg')
    analysis = json.loads((base / 'large_error_analysis.json').read_text())
    rows = json.loads((base / 'window_estimates.json').read_text())['30']
    good = [r for r in rows if r['assessable'] and abs(r['estimates']['mean_spectrum'] - r['reference_bpm']) <= 2]
    dataset = WireBreathDataset('data/WiReBreath', window_seconds=30, stride_seconds=30, normalize=False)
    indices = {row['sample_id']: i for i, row in enumerate(rows)}
    cases, control_evidence = [], {}
    for number, case in enumerate(analysis['large_cases'], 1):
        control, tier = select_control(case, good)
        sample = dataset[indices[case['sample_id']]]
        control_sample = dataset[indices[control['sample_id']]]
        evidence, _ = waveform_evidence(sample, case['estimate_bpm'], case['reference_bpm'])
        if control['sample_id'] not in control_evidence:
            control_evidence[control['sample_id']], _ = waveform_evidence(control_sample, control['estimates']['mean_spectrum'], control['reference_bpm'])
        filename = f'case_{number:02d}.png'
        plot_pair(case, control, sample, control_sample, output / filename)
        cases.append({'sample_id': case['sample_id'], 'recording_id': case['recording_id'],
                      'error_type': case['error_type'], 'error_bpm': case['error'],
                      'estimate_bpm': case['estimate_bpm'], 'reference_bpm': case['reference_bpm'],
                      'control_id': control['sample_id'], 'control_tier': tier, 'figure': filename, **evidence})
    report = {'cases': cases, 'unique_controls': control_evidence,
              'case_tags': dict(Counter(tag for row in cases for tag in row['tags'])),
              'control_tags': dict(Counter(tag for row in control_evidence.values() for tag in row['tags'])),
              'control_tiers': dict(Counter(row['control_tier'] for row in cases)),
              'limitations': ['Tags overlap and are diagnostic heuristics, not established causes.',
                             'Peak count uses fixed filtering/prominence; it is not independent ground truth.',
                             'Reference short-window spectral changes can reflect limited resolution rather than real breathing changes.',
                             'Control matches reuse some windows and are not independent replicated experiments.']}
    (output / 'review.json').write_text(json.dumps(report, indent=2))
    entries = ''.join(f'<h2>Case {i}: {row["error_type"]}, error {row["error_bpm"]:.2f} BPM</h2><p>{row["sample_id"]}</p><p>{", ".join(row["tags"])}</p><img src="{row["figure"]}" style="width:100%;max-width:1600px">' for i, row in enumerate(cases, 1))
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Respiratory rate waveform review</title><h1>35 error cases with matched accurate controls</h1>' + entries)
    print(json.dumps({k: v for k, v in report.items() if k not in ('cases', 'unique_controls')}, indent=2))
    print('Wrote 35 paired figures and review.json to', output)


if __name__ == '__main__':
    main()
