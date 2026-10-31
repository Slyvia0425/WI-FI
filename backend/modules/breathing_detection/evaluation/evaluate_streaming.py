"""Drive the streaming pipeline from independent real recordings, no refitting."""
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

from ..inference.frozen_detector import FrozenBreathingDetector
from ..inference.streaming import BreathingStream, StreamConfig
from ..preprocessing.unified_rate import ESTIMATOR_SPEC
from ..preprocessing.wire_breath import WireBreathDataset
from .evaluate_lstm import checkpoint_sha256


def summary(sequences):
    windows = [event for seq in sequences for event in seq if 'status' in event]
    valid = [e for e in windows if e['status'] != 'not_assessable']
    scores = [e for e in valid if e['temporal_instability'] is not None]
    flagged = [e for e in scores if e['temporal_instability'] >= .8]
    flips, transitions = 0, 0
    for sequence in sequences:
        previous = None
        for e in sequence:
            if 'status' not in e or e['status'] == 'not_assessable':
                previous = None
                continue
            if previous is not None:
                transitions += 1
                flips += previous != e['status']
            previous = e['status']
    return {'windows': len(windows), 'assessable': len(valid),
            'coverage': len(valid) / len(windows) if windows else None,
            'breathing_recall': sum(e['status'] == 'breathing' for e in valid) / len(valid) if valid else None,
            'false_none': sum(e['status'] == 'none' for e in valid),
            'valid_adjacent_pairs': transitions, 'label_flips': flips,
            'flip_fraction': flips / transitions if transitions else None,
            'instability_available': len(scores), 'diagnostic_threshold': .8,
            'diagnostic_flagged': len(flagged),
            'diagnostic_false_none_rejected': sum(e['status'] == 'none' for e in flagged),
            'diagnostic_correct_breathing_rejected': sum(e['status'] == 'breathing' for e in flagged),
            'experimental_rate_outputs': sum(e['rate_bpm'] is not None for e in windows)}


def main():
    output = Path('backend/modules/breathing_detection/outputs/streaming_validation_unified_v1')
    output.mkdir(exist_ok=True)
    if (output / 'config.json').exists():
        raise FileExistsError('Preserve configured streaming experiment')
    torch.set_num_threads(4)
    checkpoint = Path('backend/modules/breathing_detection/outputs/lstm_interpolated_full_seed123/best_model.pt')
    detector = FrozenBreathingDetector(checkpoint)
    config = StreamConfig(breathing_threshold=detector.threshold)
    (output / 'config.json').write_text(json.dumps({'stream': asdict(config),
        'checkpoint_sha256': checkpoint_sha256(checkpoint), 'delivery_interval_seconds': .25,
        'rate_estimator': ESTIMATOR_SPEC,
        'arrival_clock': 'virtual 250 ms delivery schedule, explicitly injected; poll even for empty intervals',
        'segment_boundary': 'new stream per MAT file, never concatenate segments',
        'temporal_gate_enabled': False, 'threshold_selection': 'none; .8 diagnostic only'}, indent=2))
    source = WireBreathDataset('data/WiReBreath', window_seconds=30, stride_seconds=30, normalize=False)
    records = []
    for index, reference in enumerate(source._windows):
        payload = loadmat(reference.path, squeeze_me=True, variable_names=('time', 'csi_abs'))
        t = np.asarray(payload['time'], dtype=float).reshape(-1)
        x = np.asarray(payload['csi_abs'], dtype=np.float32)
        stream = BreathingStream(detector, config)
        events = []
        start = 0
        final_clock = float(np.ceil((t[-1] - t[0]) / .25) * .25)
        for clock in np.arange(0., final_clock + .125, .25):
            end = int(np.searchsorted(t, t[0] + clock, side='right'))
            if end > start:
                events.extend(stream.push(t[start:end], x[start:end], arrival_time=float(clock)))
                start = end
            else:
                stream.poll(float(clock))
        active = stream.poll(final_clock)
        stale = stream.poll(final_clock + config.disconnect_seconds)
        assert stale['stream_state'] == 'disconnected' and stale['rate_bpm'] is None
        records.append({'sample_id': reference.sample_id, 'recording_id': reference.recording_id,
                        'participant_id': reference.metadata['participant_id'],
                        'environment': reference.metadata['environment'], 'events': events,
                        'active_final': active, 'after_timeout': stale})
        if (index + 1) % 100 == 0:
            print('Processed', index + 1, 'independent segments', flush=True)
    sequences = [r['events'] for r in records]
    nonoverlap = [[e for e in seq if 'window_end' in e and np.isclose(e['window_end'] % 10, 0)] for seq in sequences]
    report = {'segments': len(records), 'source_recordings': len({r['recording_id'] for r in records}),
              'five_second_updates': summary(sequences), 'ten_second_subset': summary(nonoverlap),
              'by_participant': {p: summary([r['events'] for r in records if r['participant_id'] == p]) for p in sorted({r['participant_id'] for r in records})},
              'resets': dict(Counter(e['reason'] for seq in sequences for e in seq if e.get('event') == 'reset')),
              'watchdog_stale_checks_passed': len(records),
              'limitations': ['Only ~30 seconds per independent file; not long-duration continuous validation.',
                             'Overlapping windows are correlated and remain grouped by source recording.',
                             'All data are breathing-present; no specificity or binary Macro-F1 is measurable.',
                             'Ten-second subset uses the same 5-second tracker history; do not compare its risk flags as an independent tracker.',
                             'Arrival schedule is simulated, not measured network packet jitter.',
                             'No clinical pause/motion ground truth or temporal-gate promotion.']}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2))
    (output / 'stream_events.json').write_text(json.dumps(records, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
