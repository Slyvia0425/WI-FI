"""CSI-only signal-pollution proxies, not labelled physical-motion detection."""
from collections import Counter
import numpy as np
from scipy.signal import detrend
from .respiratory_rate import spectra

FEATURES = ('jump_ratio', 'energy_burst_ratio', 'broadband_fraction', 'adjacent_incoherence')


def pollution_features(x, rate_hz=30.):
    """Extract scale-normalized features from exactly one 30s amplitude window."""
    values = np.asarray(x, dtype=float)
    if (values.ndim != 3 or values.shape[0] != 1 or values.shape[1] < 2
            or values.shape[-1] != 900 or rate_hz != 30. or not np.isfinite(values).all()):
        raise ValueError('Require finite [1,K>=2,900] amplitudes at 30 Hz')
    centered = detrend(values[0], axis=1, type='linear')
    scale = centered.std(axis=1)
    valid = scale > 1e-8
    adjacent_valid = valid[:-1] & valid[1:]
    if valid.sum() < 2 or not adjacent_valid.any():
        raise ValueError('Need at least one physically adjacent nonconstant subcarrier pair')
    z = centered/np.maximum(scale[:, None], 1e-8)
    differences = abs(np.diff(z[valid], axis=1))
    jumps = differences.max(axis=1)/np.maximum(np.median(differences, axis=1), 1e-6)
    block_rms = np.sqrt(np.mean(z[valid].reshape(-1,30,30)**2, axis=2))
    bursts = block_rms.max(axis=1)/np.maximum(np.median(block_rms, axis=1), 1e-6)
    frequencies, power = spectra(values[0], rate_hz, band_hz=(.1,5.))
    fraction = power[:, frequencies > .7].sum(axis=1)/np.maximum(power.sum(axis=1),1e-12)
    correlation = np.mean(z[:-1]*z[1:], axis=1)[adjacent_valid]
    return {'available': True,
            'jump_ratio': float(np.quantile(jumps,.9)),
            'energy_burst_ratio': float(np.quantile(bursts,.9)),
            'broadband_fraction': float(np.mean(fraction)),
            'adjacent_incoherence': float(np.clip(1-np.median(abs(correlation)),0,1)),
            'valid_subcarriers': int(valid.sum()), 'valid_adjacent_pairs': int(adjacent_valid.sum())}


def pollution_rules():
    return [dict(name=f'{family}_q{int(q*100)}', family=family, quantile=q)
            for q in (.90,.95) for family in ('transient','broadband','either_pair','three_of_four')]


def weighted_quantile(values, weights, q):
    values, weights = np.asarray(values,dtype=float), np.asarray(weights,dtype=float)
    if (values.ndim != 1 or values.shape != weights.shape or not len(values)
            or not np.isfinite(values).all() or not np.isfinite(weights).all()
            or np.any(weights <= 0) or not 0 <= q <= 1):
        raise ValueError('Require finite aligned values, positive weights and q in [0,1]')
    order = np.argsort(values, kind='stable')
    cdf = np.cumsum(weights[order])/weights.sum()
    return float(values[order[min(int(np.searchsorted(cdf,q,side='left')),len(values)-1)]])


def calibrate_thresholds(training_rows, features, q):
    """Only training IDs and CSI features enter calibration; no errors/labels."""
    rows = [r for r in training_rows if features[r['sample_id']].get('available',False)]
    if not rows:
        raise ValueError('No feature-available training rows')
    counts = Counter(r['recording_id'] for r in rows)
    weights = [1/counts[r['recording_id']] for r in rows]
    return {key: weighted_quantile([features[r['sample_id']][key] for r in rows],weights,q) for key in FEATURES}


def pollution_withhold(features, rule, thresholds):
    if rule is None or not features.get('available',False):
        return False
    # Strictly above threshold: a constant feature never flags everything.
    high = {key: features[key] > thresholds[key] for key in FEATURES}
    transient = high['jump_ratio'] and high['energy_burst_ratio']
    broadband = high['broadband_fraction'] and high['adjacent_incoherence']
    if rule['family'] == 'transient':
        return transient
    if rule['family'] == 'broadband':
        return broadband
    if rule['family'] == 'either_pair':
        return transient or broadband
    if rule['family'] == 'three_of_four':
        return sum(high.values()) >= 3
    raise ValueError('Unknown pollution-rule family')
