"""Local-peak helpers shared with the fixed estimator and historical experiments."""
import numpy as np
from scipy.signal import find_peaks

from .respiratory_rate import spectra


def local_candidates(x, rate_hz=30.):
    # Inspect a wider spectrum so a true peak near 0.1 Hz is not removed
    # merely because it lies near the final allowed respiratory boundary.
    f, power = spectra(np.asarray(x)[0], rate_hz, band_hz=(.05, 1.))
    mean = power.mean(axis=0)
    peaks, properties = find_peaks(mean, prominence=0.)
    scale = max(float(mean[(f >= .1) & (f <= .7)].max()), 1e-12)
    return [dict(bpm=float(f[p] * 60), height=float(mean[p] / scale),
                 prominence=float(properties['prominences'][j] / scale))
            for j, p in enumerate(peaks) if .1 <= f[p] <= .7]


def choose_peak(peaks, ranking='height', minimum_prominence=0.):
    valid = [p for p in peaks if p['prominence'] >= minimum_prominence]
    if not valid:
        return None
    return max(valid, key=lambda p: (p[ranking], p['height'], -p['bpm']))['bpm']


def search_endpoints(time_points=900, rate_hz=30.):
    nfft = 2 ** int(np.ceil(np.log2(4 * time_points)))
    f = np.fft.rfftfreq(nfft, 1 / rate_hz)
    allowed = f[(f >= .1) & (f <= .7)] * 60
    return float(allowed[0]), float(allowed[-1])


def conservative_boundary_rate(baseline, peaks, minimum_prominence=0., endpoints=None):
    """Only reconsider an exact search endpoint; preserve interior/no-peak cases."""
    low, high = search_endpoints() if endpoints is None else endpoints
    if not (abs(baseline - low) <= 1e-6 or abs(baseline - high) <= 1e-6):
        return baseline
    candidate = choose_peak(peaks, ranking='height', minimum_prominence=minimum_prominence)
    return baseline if candidate is None else candidate
