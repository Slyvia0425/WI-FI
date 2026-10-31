"""CSI-only comparison of frozen original/common/relative rate evidence."""
import math

TOLERANCE_BPM = 2.
ROUTES = ('baseline', 'common', 'relative')


def match_peak(peaks, target):
    matches = [(i+1, peak) for i, peak in enumerate(peaks) if abs(peak['bpm']-target) <= TOLERANCE_BPM]
    if not matches:
        return None
    rank, peak = max(matches, key=lambda pair: (pair[1]['height'], -abs(pair[1]['bpm']-target), -pair[1]['bpm']))
    return {'rank': rank, **{key: float(peak[key]) for key in ('bpm', 'height', 'prominence')}}


def three_route_features(components):
    """Accept component estimates/CSI peaks only; never reference diagnostics."""
    rates, peaks = {}, {}
    for name in ROUTES:
        component = components[name]
        if not component['available']:
            rates[name], peaks[name] = None, []
            continue
        rate = float(component['estimate']['bpm'])
        if not math.isfinite(rate) or not 6 <= rate <= 42:
            raise ValueError('Invalid component BPM')
        candidate_peaks = component['evidence']['peaks']
        for peak in candidate_peaks:
            if any(not math.isfinite(float(peak[key])) for key in ('bpm', 'height', 'prominence')):
                raise ValueError('Nonfinite candidate evidence')
            if not 6 <= peak['bpm'] <= 42 or not 0 <= peak['height'] <= 1.+1e-8 or peak['prominence'] < 0:
                raise ValueError('Invalid candidate evidence')
        rates[name], peaks[name] = rate, candidate_peaks
    if rates['baseline'] is None:
        raise ValueError('Original estimate is required')
    result = {'rates': rates, 'all_available': all(rate is not None for rate in rates.values())}
    for left, right in (('baseline', 'relative'), ('baseline', 'common'), ('common', 'relative')):
        distance = abs(rates[left]-rates[right]) if rates[left] is not None and rates[right] is not None else None
        result[f'{left}_{right}_distance_bpm'] = distance
        result[f'{left}_{right}_agree'] = distance is not None and distance <= TOLERANCE_BPM
    result['proposes_change'] = rates['relative'] is not None and result['baseline_relative_distance_bpm'] > TOLERANCE_BPM
    result['matched_peaks'] = {name: {target: match_peak(peaks[name], rates[target]) if rates[target] is not None else None
                                    for target in ('baseline', 'relative')} for name in ROUTES}
    return result
