"""Fixed development-reference BPM estimator shared by offline and stream paths."""
import numpy as np

from .respiratory_rate import estimate_rates
from .rate_peaks import conservative_boundary_rate, local_candidates, search_endpoints

ESTIMATOR_ID = 'mean_spectrum_boundary0_v1'
ESTIMATOR_SPEC = {
    'id': ESTIMATOR_ID, 'status': 'experimental_development_reference',
    'window_seconds': 30, 'sampling_rate_hz': 30, 'time_points': 900,
    'band_hz': [.1, .7], 'peak_context_hz': [.05, 1.],
    'minimum_prominence': 0., 'ranking': 'height',
    'rule': 'exact search endpoint only; preserve interior and no-local-peak outputs',
    'reliability_gate_enabled': False, 'temporal_gate_enabled': False,
}


def estimate_unified_rate(x, rate_hz=30.):
    """Estimate from one independent [1,K,900] amplitude window, no labels/IDs.

    Quality and breathing-evidence checks belong to the caller. Returning a BPM
    or preserving a boundary fallback never certifies physiological reliability.
    """
    values = np.asarray(x)
    if (values.ndim != 3 or values.shape[0] != 1 or values.shape[1] < 1
            or values.shape[-1] != 900 or rate_hz != 30. or not np.isfinite(values).all()):
        raise ValueError('Unified rate requires finite [1,K,900] CSI at exactly 30 Hz')
    baseline = estimate_rates(values, rate_hz)['mean_spectrum']
    endpoints = search_endpoints(900, rate_hz)
    boundary = any(abs(baseline - endpoint) <= 1e-6 for endpoint in endpoints)
    peaks = local_candidates(values, rate_hz) if boundary else []
    bpm = conservative_boundary_rate(baseline, peaks, minimum_prominence=0., endpoints=endpoints)
    changed = abs(bpm-baseline) > 1e-6
    return {'bpm': float(bpm), 'baseline_bpm': float(baseline), 'estimator_id': ESTIMATOR_ID,
            'correction_applied': changed,
            'reason': 'boundary_corrected' if changed else 'boundary_no_local_peak' if boundary and not peaks
                      else 'boundary_local_peak_unchanged' if boundary else 'interior_preserved',
            'reliability_gate_enabled': False}
