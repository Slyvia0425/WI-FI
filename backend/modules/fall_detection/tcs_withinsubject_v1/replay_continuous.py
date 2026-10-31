"""Frozen-checkpoint, fixed-threshold replay on audited TCS continuous recordings.

This is an evaluation utility only. It does not train, tune, or rewrite the model.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import signal

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent
sys.path.insert(0, str(MODULE / 'tcs_fall_audit_v1'))
sys.path.insert(0, str(HERE))
from audit_local_archives import canonical_iq, fields, rf_scaled_amplitude
from model import CNN
from prepare import sha

DEFAULT_ZIP = Path('/Users/chenxiaohe/Downloads/1.zip')
DEFAULT_CKPT = MODULE / 'outputs/tcs_subject01_gcv_cnn_seed42_v1/run/best.pt'
DEFAULT_RECORDS = MODULE / 'reports/tcs_fall_local_v1/records.json'
DEFAULT_AUDIT = MODULE / 'reports/tcs_fall_local_v1/audit.json'
WINDOW = 8000
STRIDE = 1000
TOLERANCE = 3


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2))


def decode_amplitude(data):
    if len(data) % 575:
        raise ValueError('Continuous record byte count is not a whole number of 575-byte packets')
    records = np.frombuffer(data, dtype=np.uint8).reshape(-1, 575)
    if len(records) < WINDOW:
        raise ValueError('Continuous record is shorter than the model window')
    valid = ((records[:, 0] == 2) & (records[:, 1] == 61) & (records[:, 2] == 187)
             & (records[:, 11] == 3) & (records[:, 12] == 3)
             & (records[:, 19] == 40) & (records[:, 20] == 2))
    if not valid.all():
        raise ValueError(f'Unexpected packet layout in {int((~valid).sum())} packets')
    raw_ts, _, _ = fields(records)
    dt = np.diff(raw_ts)
    dt = np.where(dt < -(2**31), dt + 2**32, dt)
    if (dt <= 0).any():
        raise ValueError('Nonincreasing packet timestamp')
    elapsed = np.concatenate(([0.0], np.cumsum(dt, dtype=np.float64) / 1e6))
    amp_parts = []
    for start in range(0, len(records), 1024):
        block = records[start:start + 1024]
        _, _, permutation = fields(block)
        if not np.all(np.sort(permutation, axis=1) == np.array([0, 1, 2])):
            raise ValueError('Invalid Rx antenna permutation')
        iq = canonical_iq(block, permutation)
        amp, zero_power, zero_rssi = rf_scaled_amplitude(iq, block)
        if zero_power or zero_rssi or not np.isfinite(amp).all():
            raise ValueError('RF-scaled amplitude is undefined or non-finite')
        amp_parts.append(amp[:, 0].reshape(len(block), 90))
    amplitude = np.concatenate(amp_parts).astype(np.float32, copy=False)
    return amplitude, elapsed


def make_feature(window, b, a):
    filtered = signal.filtfilt(b, a, window, axis=0)
    grouped = filtered.reshape(400, 20, 90)
    means = grouped.mean(axis=1)
    stds = grouped.std(axis=1, ddof=0)
    feature = np.divide(stds, means, out=np.zeros_like(stds), where=means != 0).astype(np.float32)
    if feature.shape != (400, 90) or not np.isfinite(feature).all():
        raise ValueError('Non-finite or unexpected feature shape')
    return feature[None]


def inference_episode(rows):
    """Positive predictions at consecutive sampled windows are one alert episode."""
    episodes = []
    active = []
    for row in rows:
        if row['prediction']:
            if active and row['window_index'] != active[-1]['window_index'] + 1:
                episodes.append(active); active = []
            active.append(row)
        elif active:
            episodes.append(active); active = []
    if active:
        episodes.append(active)
    return episodes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive', type=Path, default=DEFAULT_ZIP)
    ap.add_argument('--checkpoint', type=Path, default=DEFAULT_CKPT)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--backend', choices=('cpu', 'mps'), default='mps')
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory to preserve prior results')
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if args.backend == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS requested but unavailable; choose CPU explicitly if intended')

    audit = json.loads(DEFAULT_AUDIT.read_text())
    if sha(args.archive) != audit['source_archives'][0]['sha256']:
        raise ValueError('Raw archive SHA does not match the audited archive')
    records = json.loads(DEFAULT_RECORDS.read_text())
    continuous = sorted((r for r in records if not r.get('errors')
                        and r.get('source_kind') == 'annotated_continuous_candidate'),
                       key=lambda r: r['archive_member'])
    if len(continuous) != 12:
        raise ValueError(f'Expected 12 audited continuous records, found {len(continuous)}')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['threshold'] != 0.5:
        raise ValueError('Checkpoint threshold is not the frozen 0.5')
    if checkpoint['model_version'] != 'tcs_subject01_gcv_cnn_pilot_v1':
        raise ValueError('Unexpected checkpoint/model version')
    cfg = checkpoint['run_config']
    for name in ('model.py', 'prepare.py'):
        if sha(HERE / name) != cfg['source_sha256'][name]:
            raise ValueError(f'Frozen model/preprocessing source changed: {name}')
    if sha(MODULE / 'tcs_fall_audit_v1/audit_local_archives.py') != cfg['native_decoder_sha256']:
        raise ValueError('Native packet decoder differs from the checkpoint provenance')
    checkpoint_sha = sha(args.checkpoint)
    model = CNN().to(args.backend).eval()
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    b, a = signal.butter(8, 0.2, 'lowpass')
    all_rows, per_record = [], []
    start_all = time.perf_counter()
    with zipfile.ZipFile(args.archive) as archive:
        for rec_i, rec in enumerate(continuous, 1):
            member = rec['archive_member']
            data = archive.read(member)  # CRC is checked by zipfile on complete read.
            if hashlib.sha256(data).hexdigest() != rec['raw_file_sha256']:
                raise ValueError(f'Raw SHA mismatch: {member}')
            amp, elapsed = decode_amplitude(data)
            if len(amp) != rec['packet_count']:
                raise ValueError(f'Packet count mismatch: {member}')
            ann = rec['annotation_record']
            if not ann or ann['event_definition'] != 'recorded fall time; onset/impact meaning not established':
                raise ValueError(f'Missing or changed event annotation: {member}')
            fall_time = int(ann['recorded_fall_time_seconds'])
            starts = np.arange(0, len(amp) - WINDOW + 1, STRIDE, dtype=np.int64)
            rows = []
            chunk_size = 16
            for batch_start in range(0, len(starts), chunk_size):
                features, metadata = [], []
                for local_index, packet_start in enumerate(starts[batch_start:batch_start + chunk_size], batch_start):
                    packet_end = int(packet_start) + WINDOW - 1
                    feat = make_feature(amp[int(packet_start):packet_end + 1], b, a)
                    features.append(feat)
                    metadata.append((local_index, int(packet_start), packet_end,
                                     float(elapsed[packet_end]),
                                     float(elapsed[packet_end] - elapsed[int(packet_start)])))
                # Each item is [channel=1, time=400, feature=90]; stack adds batch axis.
                batch = torch.from_numpy(np.stack(features, axis=0)).to(args.backend)
                with torch.inference_mode():
                    scores = model(batch).softmax(1)[:, 1].detach().cpu().numpy()
                for (win_i, packet_start, packet_end, end_s, span_s), score in zip(metadata, scores):
                    row = {
                        'member': member, 'activity_id': rec['activity_id'], 'repeat_id': rec['repeat_id'],
                        'window_index': win_i, 'start_packet': packet_start, 'end_packet': packet_end,
                        'window_start_elapsed_seconds': float(elapsed[packet_start]),
                        'window_end_elapsed_seconds': end_s, 'actual_window_span_seconds': span_s,
                        'fall_annotation_seconds': fall_time, 'fall_score': float(score),
                        'threshold': 0.5, 'prediction': bool(score >= 0.5),
                        'official_event_interval_start_seconds': fall_time - TOLERANCE,
                        'official_event_interval_end_seconds': fall_time + TOLERANCE + 8,
                    }
                    row['within_official_event_interval'] = bool(
                        row['official_event_interval_start_seconds'] <= end_s <= row['official_event_interval_end_seconds'])
                    rows.append(row)
            episodes = inference_episode(rows)
            valid_alerts = [r for r in rows if r['prediction'] and r['within_official_event_interval']]
            event_detected = bool(valid_alerts)
            latency = valid_alerts[0]['window_end_elapsed_seconds'] - fall_time if event_detected else None
            true_episodes = [ep for ep in episodes if any(r['within_official_event_interval'] for r in ep)]
            false_episodes = [ep for ep in episodes if not any(r['within_official_event_interval'] for r in ep)]
            first_end, last_end = rows[0]['window_end_elapsed_seconds'], rows[-1]['window_end_elapsed_seconds']
            monitor_seconds = max(0.0, last_end - first_end)
            event_start, event_end = fall_time - TOLERANCE, fall_time + TOLERANCE + 8
            overlap = max(0.0, min(last_end, event_end) - max(first_end, event_start))
            background_seconds = max(0.0, monitor_seconds - overlap)
            record_result = {
                'member': member, 'activity_id': rec['activity_id'], 'repeat_id': rec['repeat_id'],
                'annotation_fall_seconds': fall_time, 'annotation_duration_seconds': ann['recorded_duration_seconds'],
                'annotation_semantics': ann['event_definition'], 'packet_count': len(amp),
                'raw_timestamp_span_seconds': float(elapsed[-1]),
                'window_count': len(rows), 'window_seconds_packet_positions': WINDOW,
                'window_actual_span_range_seconds': [min(r['actual_window_span_seconds'] for r in rows),
                                                     max(r['actual_window_span_seconds'] for r in rows)],
                'stride_packet_positions': STRIDE, 'stride_actual_seconds_median': float(np.median(np.diff([r['window_end_elapsed_seconds'] for r in rows]))),
                'first_window_end_elapsed_seconds': first_end, 'last_window_end_elapsed_seconds': last_end,
                'event_detected_official_tolerance': event_detected,
                'first_valid_alert_end_elapsed_seconds': valid_alerts[0]['window_end_elapsed_seconds'] if event_detected else None,
                'nominal_event_to_alert_latency_seconds': latency,
                'positive_windows_total': sum(r['prediction'] for r in rows),
                'positive_windows_inside_event_interval': sum(r['prediction'] and r['within_official_event_interval'] for r in rows),
                'positive_windows_outside_event_interval': sum(r['prediction'] and not r['within_official_event_interval'] for r in rows),
                'alarm_episodes_total': len(episodes), 'event_episodes': len(true_episodes),
                'false_alarm_episodes': len(false_episodes),
                'eligible_monitor_seconds': monitor_seconds,
                'event_interval_excluded_from_background_seconds': overlap,
                'non_event_monitor_seconds_for_false_alarm_rate': background_seconds,
                'false_alarms_per_hour': len(false_episodes) / (background_seconds / 3600) if background_seconds else None,
                'max_score': float(max(r['fall_score'] for r in rows)),
                'max_score_elapsed_seconds': rows[int(np.argmax([r['fall_score'] for r in rows]))]['window_end_elapsed_seconds'],
            }
            per_record.append(record_result)
            all_rows.extend(rows)
            print(json.dumps({'record': member, 'windows': len(rows), 'event_detected': event_detected,
                              'latency_nominal_s': latency, 'false_alarm_episodes': len(false_episodes),
                              'elapsed_s': round(time.perf_counter() - start_all, 1)}), flush=True)
            del data, amp, elapsed

    # Persist raw scores first; all downstream statistics are reproducible from these rows.
    prediction_path = args.output / 'window_predictions.csv'
    with prediction_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    monitored = sum(r['eligible_monitor_seconds'] for r in per_record)
    background = sum(r['non_event_monitor_seconds_for_false_alarm_rate'] for r in per_record)
    total_fa = sum(r['false_alarm_episodes'] for r in per_record)
    detected = sum(r['event_detected_official_tolerance'] for r in per_record)
    latencies = [r['nominal_event_to_alert_latency_seconds'] for r in per_record
                 if r['nominal_event_to_alert_latency_seconds'] is not None]
    summary = {
        'protocol': {
            'model_version': checkpoint['model_version'], 'checkpoint': str(args.checkpoint.resolve()),
            'checkpoint_sha256': checkpoint_sha, 'threshold': 0.5, 'threshold_source': 'saved checkpoint; unchanged',
            'backend': args.backend, 'record_count': len(per_record),
            'window': '8000 packet positions, same packet-count length as SA training',
            'stride': '1000 packet positions (nominal about 1 second; timestamps determine reported times)',
            'event_match': 'Official TCS-Fall code: a positive window is a hit when its endpoint is within [fall_time-3, fall_time+11] seconds.',
            'alert_episode': 'One episode is a maximal run of positive predictions at consecutive sampled windows; no debounce, hold, or extra threshold.',
            'false_alarm_rate_denominator': 'Eligible output timeline from first to last window end, minus the official event matching interval for each recording.',
            'latency': 'First in-tolerance positive window end timestamp minus integer-second workbook event time; nominal only because event time origin/onset/impact are unverified.',
            'preprocessing': 'Frozen training transform: Tx0 RF-scaled amplitude, Butterworth(8,0.2) filtfilt separately over each available 8000-packet trailing window, GCV over 20-packet groups. No future packets beyond window end are used; full 8-second buffer is required.',
            'limitations': ['One nominal subject only; no cross-subject evaluation.',
                            '12 recordings are repeated scripted trials, not independent people/events.',
                            'Workbook event time is integer seconds; its exact alignment and whether it means onset or impact are unverified.',
                            'False alarms/hour is exploratory under the stated merging/denominator convention, not an official paper metric.',
                            'Alert latency is source timestamp to model score only; device computation, network, and notification delivery are excluded.',
                            'Overlapping windows are highly correlated; window counts are not independent sample counts.',
                            'The archive still has three damaged unrelated SA members; all 12 continuous recordings are individually CRC/SHA verified.'],
        },
        'metrics': {
            'events_detected': detected, 'events_total': len(per_record),
            'event_recall': detected / len(per_record),
            'false_alarm_episodes': total_fa,
            'non_event_monitor_hours': background / 3600,
            'false_alarms_per_hour': total_fa / (background / 3600) if background else None,
            'median_nominal_event_to_alert_latency_seconds': float(np.median(latencies)) if latencies else None,
            'mean_nominal_event_to_alert_latency_seconds': float(np.mean(latencies)) if latencies else None,
            'latency_detected_events_only_seconds': latencies,
            'eligible_monitor_hours_including_event_interval': monitored / 3600,
            'positive_windows': sum(r['positive_windows_total'] for r in per_record),
            'positive_windows_outside_event_interval': sum(r['positive_windows_outside_event_interval'] for r in per_record),
        },
        'per_record': per_record,
        'archive_sha256': audit['source_archives'][0]['sha256'],
        'archive_integrity_status': 'source ZIP has 3 unrelated damaged SA entries; each replayed continuous DAT passed zip CRC and audited raw SHA',
        'elapsed_wall_seconds': time.perf_counter() - start_all,
        'training_performed': False, 'threshold_changed': False,
    }
    write_json(args.output / 'replay_summary.json', summary)
    plot_path = args.output / 'continuous_scores.png'
    fig, axes = plt.subplots(4, 3, figsize=(17, 12), sharex=False, sharey=True)
    for ax, rec in zip(axes.flat, per_record):
        rows = [r for r in all_rows if r['member'] == rec['member']]
        x = [r['window_end_elapsed_seconds'] for r in rows]
        y = [r['fall_score'] for r in rows]
        ax.plot(x, y, lw=0.9)
        ax.axhline(0.5, color='black', ls='--', lw=0.8, label='threshold 0.5')
        ax.axvspan(rec['annotation_fall_seconds'] - 3, rec['annotation_fall_seconds'] + 11,
                   color='orange', alpha=0.18, label='official event tolerance')
        ax.set_title(f"activity {rec['activity_id']} repeat {rec['repeat_id']} | detected {int(rec['event_detected_official_tolerance'])}/1")
        ax.set_xlabel('Elapsed CSI time (s)'); ax.set_ylabel('Fall score')
        ax.grid(alpha=0.2)
    fig.suptitle('TCS-Fall continuous replay | frozen CNN, fixed threshold 0.5', fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(plot_path, dpi=160); plt.close(fig)
    print(json.dumps({'summary': summary['metrics'], 'elapsed_wall_seconds': summary['elapsed_wall_seconds'],
                      'output': str(args.output.resolve())}, indent=2), flush=True)


if __name__ == '__main__':
    main()
