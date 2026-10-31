"""Fixed diagnostic phase transforms; not certified physical CSI calibration."""
import numpy as np

from .isolated_spikes import resample_amplitudes

PHASE_METHODS = ('phase_raw', 'phase_adjacent', 'phase_index_residual')


def phase_source_audit(h):
    h = np.asarray(h)
    if h.ndim != 2 or h.shape[0] < 2 or h.shape[1] < 2 or not np.iscomplexobj(h):
        raise ValueError('Expected complex [native_time,ordered_carriers]')
    finite = np.isfinite(h.real) & np.isfinite(h.imag)
    zero = np.abs(h) <= 1e-12
    valid_cells = finite & ~zero
    valid = valid_cells.all(axis=0)
    indices = np.flatnonzero(valid)
    pairs = np.flatnonzero(valid[:-1] & valid[1:])
    stats = {'shape': list(h.shape), 'dtype': str(h.dtype), 'all_finite': bool(finite.all()),
             'nonzero_imaginary_fraction': float(np.mean(h.imag != 0)),
             'zero_coefficients': int(zero.sum()), 'wholly_zero_carriers': int(zero.all(axis=0).sum()),
             'intermittent_invalid_carriers': int(np.sum(~valid & ~zero.all(axis=0))),
             'valid_carriers': len(indices), 'valid_column_indices': indices.tolist(),
             'valid_adjacent_pairs': len(pairs), 'valid_pair_left_indices': pairs.tolist()}
    if len(indices):
        selected = np.asarray(h[:, indices], dtype=np.complex128)
        unit = selected/np.abs(selected)
        increments = np.angle(unit[1:]*np.conj(unit[:-1]))
        coherence = abs(np.mean(np.exp(1j*increments), axis=1))
        stats.update(temporal_wrapped_increment_abs_median=float(np.median(abs(increments))),
                     temporal_wrapped_increment_gt_pi_over2_fraction=float(np.mean(abs(increments) > np.pi/2)),
                     temporal_increment_coherence_median=float(np.median(coherence)),
                     temporal_increment_coherence_gt_point9_fraction=float(np.mean(coherence > .9)))
    if len(pairs):
        high = np.asarray(h[:, pairs+1], dtype=np.complex128)
        low = np.asarray(h[:, pairs], dtype=np.complex128)
        difference = np.angle(high*np.conj(low))
        stats['adjacent_difference_coherence_median'] = float(np.median(abs(np.mean(np.exp(1j*difference), axis=1))))
    return stats


def native_phase_traces(h):
    """Native-time traces and explicit failures; never interpolate undefined phase."""
    h = np.asarray(h)
    stats = phase_source_audit(h)
    h = np.asarray(h, dtype=np.complex128)
    indices = np.asarray(stats['valid_column_indices'], dtype=int)
    pairs = np.asarray(stats['valid_pair_left_indices'], dtype=int)
    traces, failures = {}, {}
    if len(indices):
        traces['phase_raw'] = np.unwrap(np.angle(h[:, indices]), axis=0)
    else:
        failures['phase_raw'] = 'No carriers with defined phase throughout the window'
    if len(pairs):
        traces['phase_adjacent'] = np.unwrap(np.angle(h[:, pairs+1]*np.conj(h[:, pairs])), axis=0)
    else:
        failures['phase_adjacent'] = 'No valid originally adjacent carrier pairs'
    if len(indices) >= 3:
        phase = np.unwrap(np.angle(h[:, indices]), axis=1)
        position = indices.astype(float)-indices.mean()
        slope = (phase @ position)/np.sum(position**2)
        residual = phase-phase.mean(axis=1, keepdims=True)-slope[:, None]*position[None, :]
        traces['phase_index_residual'] = np.unwrap(np.angle(np.exp(1j*residual)), axis=0)
    else:
        failures['phase_index_residual'] = 'Need at least three valid ordered carriers for index-affine projection'
    return traces, failures, stats


def phase_windows(time, h, targets):
    """Clip to current target brackets before validity/unwrap; no extra future tail."""
    time = np.asarray(time, dtype=float)
    targets = np.asarray(targets, dtype=float)
    h = np.asarray(h)
    if (time.ndim != 1 or targets.ndim != 1 or len(time) < 2 or len(targets) < 2
            or h.ndim != 2 or h.shape[0] != len(time) or not np.isfinite(time).all()
            or not np.isfinite(targets).all() or np.any(np.diff(time) <= 0) or np.any(np.diff(targets) <= 0)
            or targets[0] < time[0] or targets[-1] > time[-1]):
        raise ValueError('Expected ordered covered timestamps and matching CSI')
    lo = max(0, int(np.searchsorted(time, targets[0], side='right'))-1)
    hi = min(len(time), int(np.searchsorted(time, targets[-1], side='left'))+1)
    clipped_t, clipped_h = time[lo:hi], h[lo:hi]
    traces, failures, stats = native_phase_traces(clipped_h)
    windows = {name: resample_amplitudes(clipped_t, values, targets) for name, values in traces.items()}
    stats['native_window_points'] = len(clipped_t)
    stats['native_window_max_gap_seconds'] = float(np.diff(clipped_t).max())
    return windows, failures, stats
