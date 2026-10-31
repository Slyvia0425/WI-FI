"""Fixed equal-weight common/relative CSI decomposition; experimental only."""
import numpy as np
from scipy.signal import detrend

DECOMPOSITION_ID = 'standardized_equal_mean_residual_v1'


def decompose_common_mode(x):
    """Return common [1,1,900], relative [1,K_valid,900], and energy accounting.

    No reference, identity, carrier reordering, denoising or time concatenation.
    Common and relative components have no physical source labels.
    """
    values = np.asarray(x, dtype=np.float64)
    if (values.ndim != 3 or values.shape[0] != 1 or values.shape[1] < 2
            or values.shape[-1] != 900 or not np.isfinite(values).all()):
        raise ValueError('Expected finite [1,K>=2,900] CSI')
    centered = detrend(values[0], axis=1, type='linear')
    scales = centered.std(axis=1)
    valid = scales > 1e-8
    if valid.sum() < 2:
        raise ValueError('Need at least two nonconstant carriers')
    z = centered[valid]/scales[valid, None]
    common = z.mean(axis=0, keepdims=True)
    relative = z-common
    total_energy = float(np.mean(z*z))
    common_fraction = float(np.mean(common*common)/total_energy)
    relative_fraction = float(np.mean(relative*relative)/total_energy)
    stats = {'decomposition_id': DECOMPOSITION_ID, 'input_carriers': values.shape[1],
             'valid_carriers': int(valid.sum()), 'dropped_carriers': int((~valid).sum()),
             'common_energy_fraction': common_fraction, 'relative_energy_fraction': relative_fraction,
             'energy_sum_error': abs(common_fraction+relative_fraction-1.),
             'reconstruction_max_error': float(np.max(abs(z-(relative+common)))),
             'residual_carrier_mean_max_abs': float(np.max(abs(relative.mean(axis=0)))),
             'relative_nonflat_carriers': int(np.sum(relative.std(axis=1) > 1e-8))}
    return np.ascontiguousarray(common[None]), np.ascontiguousarray(relative[None]), stats
