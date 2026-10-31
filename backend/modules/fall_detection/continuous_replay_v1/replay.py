"""Fixed-protocol accelerated causal-endpoint replay of twelve reserved TCS recordings."""
import argparse
from collections import deque
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter_ns
import zipfile

import numpy as np
import torch
from scipy import signal

MODULE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODULE / 'tcs_withinsubject_v1'))
from model import CNN
from prepare import sha, transform_dat_bytes
sys.path.insert(0, str(MODULE / 'tcs_fall_audit_v1'))
from audit_local_archives import fields, canonical_iq, rf_scaled_amplitude


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def sync(device):
    if device == 'mps':
        torch.mps.synchronize()


def feature_from_native(amplitude, b, a):
    if amplitude.shape != (8000, 90) or not np.isfinite(amplitude).all():
        raise ValueError('Expected finite native amplitude from current/past packets')
    filtered = signal.filtfilt(b, a, amplitude, axis=0)
    grouped = filtered.reshape(400, 20, 90)
    mean = grouped.mean(axis=1)
    gcv = np.divide(grouped.std(axis=1, ddof=0), mean, out=np.zeros_like(mean), where=mean != 0)
    result = gcv.astype(np.float32)[None]
    if result.shape != (1, 400, 90) or not np.isfinite(result).all():
        raise ValueError('Nonfinite/invalid replay features')
    return result


def episode_metrics(windows, annotation):
    """No fitted rules: positive runs yield one rising-edge notification each."""
    fall = float(annotation['recorded_fall_time_seconds'])
    episodes = []
    active = None
    for row in windows:
        if row['prediction']:
            if active is None:
                active = {
                    'episode_id': len(episodes) + 1, 'first_window': row['window_index'],
                    'onset_data_time_seconds': row['window_end_seconds'],
                    'onset_simulated_completion_seconds': row['simulated_completion_seconds'],
                    'last_positive_data_time_seconds': row['window_end_seconds'],
                    'positive_windows': 0,
                }
                episodes.append(active)
            active['positive_windows'] += 1
            active['last_positive_data_time_seconds'] = row['window_end_seconds']
        else:
            active = None
    strict = [e for e in episodes if fall <= e['onset_data_time_seconds'] <= fall + 11]
    tolerant = [e for e in episodes if fall - 3 <= e['onset_data_time_seconds'] <= fall + 11]
    background = [w for w in windows if w['window_end_seconds'] <= 120]
    exposure = max(0., 120. - windows[0]['window_end_seconds'])
    background_episodes = [e for e in episodes if e['onset_data_time_seconds'] <= 120]
    positive_seconds = 0.
    for i, row in enumerate(windows):
        if row['window_end_seconds'] >= 120:
            break
        right = min(120., windows[i + 1]['window_end_seconds'] if i + 1 < len(windows) else 120.)
        if row['prediction']:
            positive_seconds += max(0., right - row['window_end_seconds'])
    contains_point = [w for w in windows if w['prediction'] and w['window_start_seconds'] <= fall <= w['window_end_seconds']]
    tolerance_overlap = [w for w in windows if w['prediction'] and
                         w['window_start_seconds'] <= fall + 3 and w['window_end_seconds'] >= fall - 3]
    proxy_false = [w for w in windows if w['prediction'] and
                   (w['window_end_seconds'] < fall - 3 or w['window_start_seconds'] > fall + 3)]
    for episode in episodes:
        time = episode['onset_data_time_seconds']
        episode['delay_from_nominal_fall_seconds'] = time - fall
        episode['category'] = ('background_false_alarm' if time <= 120 else
                               'pre_annotation_unscored_gap' if time < fall - 3 else
                               'early_annotation_tolerance_band' if time < fall else
                               'post_annotation_match_band' if time <= fall + 11 else
                               'post_event_unverified_period')
    return {
        'nominal_fall_time_seconds': fall, 'annotation_stated_duration_seconds': annotation['recorded_duration_seconds'],
        'strict_new_alert_detected': bool(strict), 'tolerance_new_alert_detected': bool(tolerant),
        'strict_data_delay_seconds': strict[0]['onset_data_time_seconds'] - fall if strict else None,
        'strict_simulated_processing_adjusted_delay_seconds': strict[0]['onset_simulated_completion_seconds'] - fall if strict else None,
        'tolerance_data_delay_seconds': tolerant[0]['onset_data_time_seconds'] - fall if tolerant else None,
        'window_contains_nominal_fall_detected': bool(contains_point),
        'first_point_containing_positive_window_delay_seconds': contains_point[0]['window_end_seconds'] - fall if contains_point else None,
        'timestamp_adapted_paper_cwr_hit': bool(tolerance_overlap),
        'timestamp_adapted_paper_newr_success': not bool(proxy_false),
        'paper_proxy_false_positive_windows': len(proxy_false),
        'total_windows': len(windows), 'positive_windows': sum(w['prediction'] for w in windows),
        'total_alarm_episodes': len(episodes), 'background_windows': len(background),
        'background_positive_windows': sum(w['prediction'] for w in background),
        'background_false_alarm_episodes': len(background_episodes),
        'background_monitoring_seconds_after_warmup': exposure,
        'background_positive_state_seconds': positive_seconds,
        'background_false_alarms_per_hour': len(background_episodes) / (exposure / 3600) if exposure else None,
        'additional_notifications_in_strict_match_band': max(0, len(strict) - 1),
        'episodes': episodes,
    }


def describe(values):
    values = [float(v) for v in values if v is not None]
    return None if not values else {'n': len(values), 'median': float(np.median(values)),
                                   'mean': float(np.mean(values)), 'p95': float(np.percentile(values, 95)),
                                   'min': min(values), 'max': max(values)}


def protect():
    old = json.loads((MODULE / 'final_selection_v1/preserved_artifacts_before.json').read_text())
    for name, digest in old.items():
        if sha(MODULE / name) != digest:
            raise ValueError('Old protected artifact changed: ' + name)
    frozen = MODULE / 'outputs/final_selection_v1/final_research_model.pt'
    receipt = json.loads(frozen.with_suffix('.json').read_text())
    if sha(frozen) != receipt['checkpoint_sha256']:
        raise ValueError('Old HP frozen model changed')
    for name, digest in receipt['inference_source_sha256'].items():
        if sha(MODULE / name) != digest:
            raise ValueError('Old HP inference source changed')
    return {'old_artifacts_checked': len(old), 'hp_frozen_model_and_source_unchanged': True}


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive', type=Path, default=Path('/Users/chenxiaohe/Downloads/1.zip'))
    ap.add_argument('--checkpoint', type=Path, default=MODULE / 'outputs/tcs_subject01_gcv_cnn_seed42_v1/run/best.pt')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--backend', choices=['mps', 'cpu'], default='mps')
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh replay directory; do not overwrite earlier evidence')
    args.output.mkdir(parents=True)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    if args.backend == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; no hidden backend switch')
    training_report = MODULE / 'reports/tcs_subject01_gcv_cnn_seed42_v1/training_summary.json'
    training = json.loads(training_report.read_text())
    checkpoint_sha = sha(args.checkpoint)
    if checkpoint_sha != training['checkpoint_sha256']:
        raise ValueError('Pilot checkpoint differs from selected training artifact')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if state['threshold'] != .5 or state['label_mapping'] != {'Nonfall': 0, 'Fall': 1}:
        raise ValueError('Unexpected frozen threshold or label mapping')
    cfg = state['run_config']
    source_dir = MODULE / 'tcs_withinsubject_v1'
    for name in ('model.py', 'prepare.py', 'train.py'):
        if sha(source_dir / name) != cfg['source_sha256'][name]:
            raise ValueError('Saved pilot source changed')
    if sha(MODULE / 'tcs_fall_audit_v1/audit_local_archives.py') != cfg['native_decoder_sha256']:
        raise ValueError('Saved native decoder changed')
    if sha(args.archive) != state['data_version']['archive_sha256']:
        raise ValueError('Archive differs from audited/training data version')
    audited = json.loads((MODULE / 'reports/tcs_fall_local_v1/records.json').read_text())
    reserved = set(state['split_identifiers']['continuous_reserved_unread'])
    records = sorted((r for r in audited if r['archive_member'] in reserved), key=lambda r: r['archive_member'])
    if len(records) != 12 or any(r['errors'] or not r['annotation_available'] for r in records):
        raise ValueError('Reserved twelve-record cohort differs from audit')
    if reserved & set(state['split_identifiers']['train_ids'] + state['split_identifiers']['validation_ids']):
        raise ValueError('Continuous/SA split overlap')
    protocol = {
        'created_utc_before_continuous_predictions': datetime.now(timezone.utc).isoformat(),
        'source_code_sha256': sha(__file__), 'checkpoint_sha256': checkpoint_sha,
        'archive': str(args.archive), 'archive_sha256': state['data_version']['archive_sha256'],
        'record_ids': [r['archive_member'] for r in records],
        'preprocess': state['preprocess_config'], 'threshold': .5,
        'input_packets': 8000, 'hop_packets': 1000,
        'window_policy': 'last8000 packets; endpoints8000,9000,...; final incomplete hop has no new score',
        'window_timestamp': 'real NIC timestamp of first and last packet, relative to first packet of recording',
        'feature_causality': 'only current/past8000 packets; filtfilt is recomputed within this available window, never over the full recording',
        'native_decoding': '1000 new packets at a time; trailing buffer capped at8000; per-packet RF scaling independent of future packets',
        'alarm_policy': 'one notification at rising edge of each contiguous positive run; reset at a negative score',
        'smoothing': None, 'confirmation_count': 1, 'cooldown_seconds': 0,
        'primary_event_match_seconds_relative_nominal_annotation': [0, 11],
        'tolerance_event_match_seconds_relative_nominal_annotation': [-3, 11],
        'matching_basis': 'fixed8s nominal context plus official3s annotation tolerance; fixed before predictions, not optimized',
        'primary_event_recall_unit': 'recording/event has at least one NEW alarm onset in match band; ongoing earlier alarms do not count as a new event alarm',
        'window_detection_secondary': 'any positive window whose actual time span contains nominal fall point',
        'false_alarm_exposure': 'documented initial120s background only, from first complete-window endpoint until120s; one rising-edge onset counts once',
        'false_alarm_counting_boundaries': 'positive windows and positive-state time also reported; no merging gap, smoothing or refractory period',
        'post120s_false_alarm_ground_truth': 'not asserted; report later unmatched notifications separately',
        'paper_proxy': 'CWR/NEWR adapted to actual timestamps and positive-window overlap with annotation+-3s; not exact official preprocessing/timeline reproduction',
        'annotation_time_zero': 'assume supplied seconds relative to first packet for nominal analysis; NOT independently synchronized; no offsets fitted',
        'main_latency': 'window-end NIC time minus nominal annotation; real device notification latency unavailable',
        'processing_adjusted_latency': 'virtual completion=max(current packet time,prior virtual completion)+measured processing; exclude archive read/CRC, startup, radio transport and UI/network notification',
        'replay': 'accelerated offline, chronological processing, batch1; no31minute wallclock sleep',
        'backend': args.backend, 'cpu_threads': 2, 'batch_size': 1, 'dtype': 'float32',
        'scope': 'same nominal subject01 as training; new continuous recordings, not cross-subject or field validation',
        'protected_before': protect(), 'training_run': False, 'threshold_changed': False,
    }
    save(args.output / 'protocol.json', protocol)
    save(args.output / 'protocol_receipt.json', {'sha256_before_predictions': sha(args.output / 'protocol.json')})
    model = CNN().to(args.backend).eval(); model.load_state_dict(state['model_state_dict'], strict=True)
    for _ in range(20):
        model(torch.zeros((1, 1, 400, 90), device=args.backend))
    sync(args.backend)
    b, a = signal.butter(8, .2, 'lowpass')
    all_windows, results, feature_checks = [], [], []
    start_run = perf_counter_ns()
    with zipfile.ZipFile(args.archive) as archive:
        for number, record in enumerate(records, 1):
            member = record['archive_member']
            data = archive.read(member)
            if hashlib.sha256(data).hexdigest() != record['raw_file_sha256']:
                raise ValueError('Continuous record differs from integrity audit: ' + member)
            native_records = np.frombuffer(data, dtype=np.uint8).reshape(-1, 575)
            timestamps, _, _ = fields(native_records)
            differences = np.diff(timestamps)
            differences = np.where(differences < -(2**31), differences + 2**32, differences)
            if (differences <= 0).any():
                raise ValueError('Nonincreasing packet timestamps')
            elapsed = np.concatenate([[0], np.cumsum(differences)]).astype(np.float64) / 1e6
            reference, _ = transform_dat_bytes(data[:8000 * 575])
            buffer = deque(maxlen=8)
            virtual_completion = 0.
            windows = []
            first_feature_error = None
            chunk_decode_times = []
            for packet_start in range(0, len(native_records), 1000):
                stop = min(packet_start + 1000, len(native_records))
                block = native_records[packet_start:stop]
                if not np.all((block[:, 0] == 2) & (block[:, 1] == 61) & (block[:, 2] == 187)
                              & (block[:, 11] == 3) & (block[:, 12] == 3)
                              & (block[:, 19] == 40) & (block[:, 20] == 2)):
                    raise ValueError('Packet layout changed')
                begin_decode = perf_counter_ns()
                _, _, permutation = fields(block)
                if not np.all(np.sort(permutation, axis=1) == np.array([0, 1, 2])):
                    raise ValueError('Invalid receive antenna permutation')
                iq = canonical_iq(block, permutation)
                scaled, undefined, zero_rssi = rf_scaled_amplitude(iq, block)
                if undefined or zero_rssi or not np.isfinite(scaled).all():
                    raise ValueError('Nonfinite RF scaling; no filling/skipping')
                amplitude = scaled[:, 0].reshape(len(block), 90)
                decode_ms = (perf_counter_ns() - begin_decode) / 1e6
                chunk_decode_times.append(decode_ms)
                available = float(elapsed[stop - 1])
                virtual_completion = max(available, virtual_completion) + decode_ms / 1000
                if len(block) < 1000:
                    continue
                buffer.append(amplitude)
                if len(buffer) < 8:
                    continue
                feature_start = perf_counter_ns()
                segment = np.concatenate(list(buffer), axis=0)
                x = feature_from_native(segment, b, a)
                feature_ms = (perf_counter_ns() - feature_start) / 1e6
                if not windows:
                    first_feature_error = float(np.abs(reference - x).max())
                    if first_feature_error > 1e-6:
                        raise ValueError('Rolling first-window features differ from saved SA recipe')
                sync(args.backend)
                inference_start = perf_counter_ns()
                output = model(torch.from_numpy(x[None]).to(args.backend))
                score = float(output.softmax(1)[0, 1].item())
                sync(args.backend)
                inference_ms = (perf_counter_ns() - inference_start) / 1e6
                if not np.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError('Invalid model score')
                virtual_completion += (feature_ms + inference_ms) / 1000
                windows.append({
                    'record_id': member, 'window_index': len(windows),
                    'start_packet': stop - 8000, 'end_packet_exclusive': stop,
                    'window_start_seconds': float(elapsed[stop - 8000]),
                    'window_end_seconds': available, 'fall_score': score,
                    'threshold': .5, 'prediction': int(score >= .5),
                    'new_packet_decode_ms': decode_ms, 'feature_ms': feature_ms,
                    'transfer_inference_score_ms': inference_ms,
                    'total_current_update_processing_ms': decode_ms + feature_ms + inference_ms,
                    'simulated_completion_seconds': virtual_completion,
                    'virtual_processing_lag_ms': (virtual_completion - available) * 1000,
                })
            if not windows:
                raise ValueError('No complete replay window')
            result = episode_metrics(windows, record['annotation_record'])
            result.update({'record_id': member, 'nominal_subject_id': record['nominal_subject_id'],
                           'activity_id': record['activity_id'], 'repeat_id': record['repeat_id'],
                           'raw_packet_count': len(native_records), 'raw_span_seconds': float(elapsed[-1]),
                           'first_window_time_seconds': windows[0]['window_end_seconds'],
                           'last_window_time_seconds': windows[-1]['window_end_seconds'],
                           'unscored_tail_packets': len(native_records) - windows[-1]['end_packet_exclusive'],
                           'first_window_feature_max_abs_difference': first_feature_error,
                           'max_virtual_processing_lag_ms': max(w['virtual_processing_lag_ms'] for w in windows),
                           'window_update_processing_ms': describe(w['total_current_update_processing_ms'] for w in windows),
                           'new_packet_decode_ms': describe(chunk_decode_times),
                           'annotation_time_origin_verified': False})
            results.append(result); all_windows.extend(windows)
            feature_checks.append({'record_id': member, 'first_window_max_abs_difference': first_feature_error,
                                   'native_shape': [8000, 90], 'model_input_shape': [1, 1, 400, 90]})
            with (args.output / (Path(member).stem + '_windows.csv')).open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(windows[0])); writer.writeheader(); writer.writerows(windows)
            save(args.output / (Path(member).stem + '_events.json'), result)
            print(json.dumps({'record': number, 'of': 12, 'id': member, 'windows': len(windows),
                              'strict_event_detected': result['strict_new_alert_detected'],
                              'window_event_detected': result['window_contains_nominal_fall_detected'],
                              'background_false_alarms': result['background_false_alarm_episodes'],
                              'strict_nominal_delay_seconds': result['strict_data_delay_seconds']}), flush=True)
            save(args.output / 'progress.json', {'completed_records': number, 'of': 12})
            del data, native_records, timestamps, elapsed, buffer
    background_seconds = sum(r['background_monitoring_seconds_after_warmup'] for r in results)
    false_episodes = sum(r['background_false_alarm_episodes'] for r in results)
    strict_count = sum(r['strict_new_alert_detected'] for r in results)
    tolerant_count = sum(r['tolerance_new_alert_detected'] for r in results)
    with (args.output / 'all_windows.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_windows[0])); writer.writeheader(); writer.writerows(all_windows)
    save(args.output / 'feature_consistency_checks.json', feature_checks)
    protocol_digest = json.loads((args.output / 'protocol_receipt.json').read_text())['sha256_before_predictions']
    if sha(args.output / 'protocol.json') != protocol_digest or sha(args.checkpoint) != checkpoint_sha:
        raise ValueError('Protocol or checkpoint changed after replay started')
    summary = {
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'completed': True,
        'checkpoint_sha256': checkpoint_sha, 'protocol_sha256': protocol_digest,
        'recordings': len(results), 'nominal_events': len(results),
        'total_full_recording_seconds': sum(r['raw_span_seconds'] for r in results),
        'windows': len(all_windows), 'strict_new_alert_events_detected': strict_count,
        'strict_new_alert_event_recall': strict_count / len(results),
        'tolerance_new_alert_events_detected': tolerant_count,
        'tolerance_new_alert_event_recall': tolerant_count / len(results),
        'nominal_point_containing_window_hits': sum(r['window_contains_nominal_fall_detected'] for r in results),
        'timestamp_adapted_paper_cwr': sum(r['timestamp_adapted_paper_cwr_hit'] for r in results) / len(results),
        'timestamp_adapted_paper_newr': sum(r['timestamp_adapted_paper_newr_success'] for r in results) / len(results),
        'background_monitoring_seconds_after_warmup': background_seconds,
        'background_false_alarm_episodes': false_episodes,
        'background_false_alarms_per_hour': false_episodes / (background_seconds / 3600),
        'background_positive_windows': sum(r['background_positive_windows'] for r in results),
        'background_windows': sum(r['background_windows'] for r in results),
        'background_positive_state_seconds': sum(r['background_positive_state_seconds'] for r in results),
        'strict_nominal_data_delay_seconds': describe(r['strict_data_delay_seconds'] for r in results),
        'strict_processing_adjusted_simulated_delay_seconds': describe(r['strict_simulated_processing_adjusted_delay_seconds'] for r in results),
        'tolerance_nominal_data_delay_seconds': describe(r['tolerance_data_delay_seconds'] for r in results),
        'local_update_processing_ms': describe(w['total_current_update_processing_ms'] for w in all_windows),
        'local_transfer_inference_score_ms': describe(w['transfer_inference_score_ms'] for w in all_windows),
        'max_virtual_processing_lag_ms': max(w['virtual_processing_lag_ms'] for w in all_windows),
        'first_window_features_match_existing_recipe': all(c['first_window_max_abs_difference'] <= 1e-6 for c in feature_checks),
        'wallclock_replay_seconds_excluding_startup': (perf_counter_ns() - start_run) / 1e9,
        'protected_after': protect(), 'training_run': False, 'threshold_changed': False,
        'threshold': .5, 'continuous_recordings_evaluated': 12,
        'physical_event_to_notification_latency_measured': False,
        'annotation_time_origin_verified': False, 'results': results,
    }
    save(args.output / 'summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'results'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
