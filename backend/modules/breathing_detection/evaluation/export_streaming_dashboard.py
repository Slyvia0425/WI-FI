"""Attach actual backend streaming results to curated frontend recordings."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat
from backend.shared.export import write_replay
from ..inference.frozen_detector import FrozenBreathingDetector
from ..inference.streaming import BreathingStream, StreamConfig
from ..preprocessing.wire_breath import WireBreathDataset
from ..preprocessing.unified_rate import ESTIMATOR_ID

ROOT = Path('backend/modules/breathing_detection/outputs')


def timeline(t, x, detector, interrupted=False):
    stream = BreathingStream(detector, StreamConfig(breathing_threshold=detector.threshold))
    snapshots, events, start = [], [], 0
    origin = float(t[0])
    # Receive the first real sample at/after 30s as the interpolation bracket.
    # Never expose a window before that packet actually arrives.
    limit = min(len(t), int(np.searchsorted(t, origin + 30., side='left')) + 1)
    for clock in np.arange(0., 34.125, .25):
        end = min(limit, int(np.searchsorted(t, origin + clock, side='right')))
        if end > start and not (interrupted and 12 <= clock < 15):
            events.extend(dict(time=float(clock), **e) for e in stream.push(t[start:end], x[start:end], arrival_time=float(clock)))
        start = end  # discard lost packets rather than replaying them on recovery
        state = stream.poll(float(clock))
        last = state.pop('last_result')
        snapshot = dict(time=float(clock), **state,
            window_start=last['window_start'] - origin if last and not state['last_result_stale'] else None,
            window_end=last['window_end'] - origin if last and not state['last_result_stale'] else None)
        # Keep only state transitions; the UI holds the last received snapshot.
        if not snapshots or any(snapshot[k] != snapshots[-1][k] for k in snapshot if k != 'time'):
            snapshots.append(snapshot)
    return dict(snapshots=snapshots, events=events, transport_gaps=[dict(start=12, end=15)] if interrupted else [])


def main():
    torch.set_num_threads(4)
    payload = json.loads((ROOT / 'replay.json').read_text())
    detector = FrozenBreathingDetector(ROOT / 'lstm_interpolated_full_seed123/best_model.pt')
    refs = {r.sample_id: r for r in WireBreathDataset('data/WiReBreath', window_seconds=30, stride_seconds=30, normalize=False)._windows}
    for record in payload['records']:
        for sample in record['samples']:
            raw = loadmat(refs[sample['id']].path, squeeze_me=True, variable_names=('time', 'csi_abs'))
            t, x = np.asarray(raw['time'], dtype=float).reshape(-1), np.asarray(raw['csi_abs'], dtype=np.float32)
            sample['streaming'] = {mode: timeline(t, x, detector, mode == 'interrupted') for mode in ('recorded', 'interrupted')}
    payload['streaming'] = dict(version=1, duration=34, signal_duration=30, update_seconds=5, watchdog_seconds=2,
        provenance='Backend BreathingStream; virtual 250ms delivery. Simulated mode discards packets at 12–15s. Extra replay time allows the final real interpolation bracket to arrive, then demonstrates 2s input timeout. No live device.',
        rate_method=ESTIMATOR_ID, reliability_gate_enabled=False)
    write_replay('breathing_detection', payload, bundle=True)
    print('Exported', sum(len(r['samples']) for r in payload['records']), 'streaming samples')


if __name__ == '__main__':
    main()
