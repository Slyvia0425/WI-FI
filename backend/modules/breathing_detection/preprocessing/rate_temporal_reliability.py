"""CSI-only within-window rate diagnostics; not a deployed reliability gate."""
import numpy as np
from scipy.signal import find_peaks

from .respiratory_rate import spectra

SUBWINDOW_SECONDS = 20
SUBWINDOW_STARTS = (0, 5, 10)
TOLERANCE_BPM = 60 / SUBWINDOW_SECONDS


def temporal_context(x, rate_hz=30.):
    """No reference/subject input: three overlapping 20s spectra and a full one."""
    values = np.asarray(x, dtype=float)
    if (values.ndim != 3 or values.shape[0] != 1 or rate_hz != 30.
            or values.shape[-1] != 900 or not np.isfinite(values).all()):
        raise ValueError('Expected finite [1,K,900] CSI at 30 Hz')
    full_f, full_power = spectra(values[0], rate_hz, band_hz=(.05, 1.))
    mean = full_power.mean(axis=0)
    peaks, _ = find_peaks(mean)
    sub_rates = []
    for start in SUBWINDOW_STARTS:
        segment = values[0, :, round(start * rate_hz):round((start + SUBWINDOW_SECONDS) * rate_hz)]
        f, power = spectra(segment, rate_hz)
        sub_rates.append(float(f[np.argmax(power.mean(axis=0))] * 60))
    return {'frequencies': full_f, 'mean_power': mean, 'peak_indices': peaks,
            'subwindow_rates_bpm': sub_rates}


def candidate_features(context, bpm):
    if not np.isfinite(bpm) or not 6 <= bpm <= 42:
        raise ValueError('Expected candidate rate in [6,42] BPM')
    sub = np.asarray(context['subwindow_rates_bpm'])
    f, mean = context['frequencies'], context['mean_power']
    hz = bpm / 60
    main = float(mean[np.argmin(abs(f - hz))])
    targets = [target for target in (hz / 2, hz * 2) if .1 <= target <= .7 and abs(target-hz) > 2/30]
    peaks = [int(i) for i in context['peak_indices'] if .1 <= f[i] <= .7 and abs(f[i]-hz) > 2/30
             and any(abs(f[i]-target) <= 1/30 for target in targets)]
    strongest = max(peaks, key=lambda i: mean[i]) if peaks else None
    harmonic_votes = sum(abs(rate-bpm) > TOLERANCE_BPM
                         and any(abs(rate-target*60) <= TOLERANCE_BPM for target in targets) for rate in sub)
    return {'available': True, 'candidate_bpm': float(bpm),
            'subwindow_rates_bpm': sub.tolist(),
            'temporal_disagreement_count': int(np.sum(abs(sub-bpm) > TOLERANCE_BPM)),
            'subwindow_spread_bpm': float(np.ptp(sub)),
            'harmonic_subwindow_votes': int(harmonic_votes),
            'harmonic_competition_ratio': float(mean[strongest] / max(main, 1e-12)) if strongest is not None else 0.,
            'harmonic_competitor_bpm': float(f[strongest]*60) if strongest is not None else None}


def reliability_rules():
    # Small fixed grid, declared before real feature extraction/outer scoring.
    return [dict(name='temporal_only', min_disagreements=3, min_ratio=None, min_harmonic_votes=None)] + [
        dict(name=f'temporal{n}_harmonic{ratio}_votes{votes}', min_disagreements=n,
             min_ratio=ratio, min_harmonic_votes=votes)
        for n in (2, 3) for ratio in (.5, 1.) for votes in (1, 2)]


def withhold_rate(features, rule):
    if rule is None or not features.get('available', False):
        return False
    if features['temporal_disagreement_count'] < rule['min_disagreements']:
        return False
    return rule['min_ratio'] is None or (
        features['harmonic_competition_ratio'] >= rule['min_ratio']
        and features['harmonic_subwindow_votes'] >= rule['min_harmonic_votes'])
