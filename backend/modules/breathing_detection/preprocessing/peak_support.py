"""Single frozen CSI-only candidate; experimental, not imported by inference."""
import numpy as np

RULE_ID = 'supported_competitor_v1'
MINIMUM_HEIGHT = .5
TOLERANCE_BPM = 2.


def supported_competitor(evidence, baseline_bpm):
    """Return a supported alternative or original, never a reference-based choice."""
    if not np.isfinite(baseline_bpm) or not 6 <= baseline_bpm <= 42:
        raise ValueError('Expected baseline in [6,42] BPM')
    carrier = np.asarray(evidence['carrier_dominant_bpm'], dtype=float)
    if carrier.ndim != 1 or not len(carrier) or not np.isfinite(carrier).all() or np.any((carrier < 6) | (carrier > 42)):
        raise ValueError('Expected finite carrier dominant rates in [6,42]')
    support = lambda bpm: float(np.mean(abs(carrier-bpm) <= TOLERANCE_BPM))
    eligible = []
    for peak in evidence['peaks']:
        bpm, height = float(peak['bpm']), float(peak['height'])
        if not np.isfinite(bpm) or not 6 <= bpm <= 42 or not np.isfinite(height) or not 0 <= height <= 1.+1e-8:
            raise ValueError('Invalid spectral candidate')
        if height >= MINIMUM_HEIGHT:
            eligible.append({'bpm': bpm, 'height': height, 'support': support(bpm)})
    current_support = support(baseline_bpm)
    winner = max(eligible, key=lambda p: (p['support'], p['height'], -p['bpm']), default=None)
    changed = winner is not None and winner['support'] > current_support and abs(winner['bpm']-baseline_bpm) > TOLERANCE_BPM
    return {'bpm': winner['bpm'] if changed else float(baseline_bpm), 'changed': bool(changed),
            'baseline_support': current_support, 'winner': winner, 'rule_id': RULE_ID,
            'reason': 'stronger_supported_competitor' if changed else 'preserve_baseline'}
