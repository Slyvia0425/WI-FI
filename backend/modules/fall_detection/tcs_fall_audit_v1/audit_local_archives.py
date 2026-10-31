"""Audit downloaded TCS-Fall archives without training or modifying prior models.

The 226 raw recordings stay inside their original ZIP. Full packet headers, CRC,
canonical IQ and RF scaling are checked in bounded blocks; one native 90-feature
recording is retained for signal inspection. No windows/labels are used to fit a model.
"""
import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import time
import zipfile

import numpy as np

MODULE = Path(__file__).resolve().parent.parent
PILOT = '1/a_01_10_00.dat'
ANNOTATION_SHA = '286d64f3925ffd92a4558e9d0c3433e6ba4bb402834243faa30bf25b3974b644'
READ_ONLY_PROTECTED = MODULE / 'final_selection_v1/preserved_artifacts_before.json'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def archive_inventory(path):
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len({entry.filename for entry in entries}) != len(entries):
            raise ValueError('Duplicate ZIP member names')
        for entry in entries:
            name = PurePosixPath(entry.filename)
            if name.is_absolute() or '..' in name.parts or entry.flag_bits & 1:
                raise ValueError('Unsafe or encrypted archive member')
        return {'file': str(path.resolve()), 'bytes': path.stat().st_size,
                'mtime_ns': path.stat().st_mtime_ns, 'sha256': sha(path),
                'entries': [{'path': e.filename, 'bytes': e.file_size, 'compressed_bytes': e.compress_size,
                             'crc32': f'{e.CRC:08x}', 'is_directory': e.is_dir()} for e in entries]}


def fields(records):
    timestamps = records[:, 3:7].astype(np.int64) @ np.array([1, 256, 65536, 16777216], dtype=np.int64)
    counter = records[:, 7:9].astype(np.int64) @ np.array([1, 256], dtype=np.int64)
    antenna = records[:, 18].astype(np.int64)
    permutation = np.stack([antenna & 3, (antenna >> 2) & 3, (antenna >> 4) & 3], axis=1)
    return timestamps, counter, permutation


def canonical_iq(records, permutation):
    # Reviewed Intel payload order: SC, Rx, Tx; 3 reserved bits precede each SC.
    bit = (np.arange(30)[:, None, None] * 147 + 3
           + np.arange(3)[None, :, None] * 48 + np.arange(3)[None, None, :] * 16).reshape(-1)
    index, shift = bit // 8, bit % 8
    payload = records[:, 23:]
    word = (payload[:, index].astype(np.uint32)
            | (payload[:, index + 1].astype(np.uint32) << 8)
            | (payload[:, index + 2].astype(np.uint32) << 16)) >> shift
    real = (word & 255).astype(np.uint8).view(np.int8)
    imag = ((word >> 8) & 255).astype(np.uint8).view(np.int8)
    iq = np.stack([real, imag], axis=-1).reshape(-1, 30, 3, 3, 2).transpose(0, 3, 2, 1, 4)
    inverse = np.argsort(permutation, axis=1)
    return np.ascontiguousarray(np.take_along_axis(iq, inverse[:, None, :, None, None], axis=2))


def rf_scaled_amplitude(iq, records):
    # Same inspected formulas as official get_scale_csi; undefined divisions stay NaN.
    real, imag = iq[..., 0].astype(np.float32), iq[..., 1].astype(np.float32)
    power = real * real + imag * imag
    per_rx_sc_power = power.sum(axis=1, dtype=np.float32) / 30
    rssi = records[:, 13:16].astype(np.float64)
    rssi_linear = np.where(rssi != 0, 10 ** (rssi / 10), 0).sum(axis=1)
    agc = records[:, 17].astype(np.float64)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        dbm = 10 * np.log10(rssi_linear) - 44 - agc
        scale = np.where(per_rx_sc_power > 0, 10 ** (dbm[:, None, None] / 10) / per_rx_sc_power, np.nan)
        noise = records[:, 16].astype(np.int16)
        noise = np.where(noise >= 128, noise - 256, noise)
        noise = np.where(noise == -127, -92, noise)
        thermal = 10 ** (noise.astype(np.float64) / 10)
        factor = np.sqrt(scale / (thermal[:, None, None] + scale * 9)) * np.sqrt(10 ** 0.45)
        # Official scaled CSI is cast back to complex64 before abs.
        csi = (real.astype(np.complex64) + 1j * imag.astype(np.complex64))
        scaled = (csi * factor[:, None, :, :]).astype(np.complex64)
        amplitude = np.abs(scaled)
    return amplitude, int((per_rx_sc_power == 0).sum()), int((rssi_linear == 0).sum())


def reference_check(source, records, permutation, sample_indices=None):
    parsed = ast.parse(source.read_text())
    wanted = {'parse_csi', 'db', 'dbinv', 'get_total_rss', 'get_scale_csi'}
    functions = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    if {n.name for n in functions} != wanted:
        raise ValueError('Reviewed reference functions changed')
    namespace = {'np': np, 'math': math}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
    maximum = 0.0
    indices = list(range(min(16, len(records)))) if sample_indices is None else list(sample_indices)
    for i in indices:
        frame = records[i, 2:].tobytes()
        original = namespace['parse_csi'](frame[21:], 3, 3)
        corrected = original.copy()
        corrected[:, permutation[i], :] = original[:, [0, 1, 2], :]
        ours_iq = canonical_iq(records[i:i + 1], permutation[i:i + 1])
        ours_complex = ours_iq[..., 0].astype(np.float32) + 1j * ours_iq[..., 1].astype(np.float32)
        if not np.array_equal(corrected, ours_complex[0]):
            raise ValueError('Native decoder or antenna permutation differs from reviewed official decoder')
        params = {'csi': corrected, 'Nrx': 3, 'Ntx': 3, 'rssi_a': frame[11],
                  'rssi_b': frame[12], 'rssi_c': frame[13], 'noise': frame[14] - 256, 'agc': frame[15]}
        expected = np.abs(namespace['get_scale_csi'](params).astype(np.complex64))
        actual, _, _ = rf_scaled_amplitude(ours_iq, records[i:i + 1])
        error = float(np.nanmax(np.abs(expected - actual[0])))
        maximum = max(maximum, error)
    if maximum > 1e-4:
        raise ValueError('RF scaling differs from official reference')
    return {'actual_packets_compared': len(indices), 'packet_indices': [int(i) for i in indices],
            'antenna_selection_codes_checked': sorted({int(records[i, 18]) for i in indices}),
            'canonical_complex_values_exact': True,
            'scaled_amplitude_max_abs_difference': maximum, 'tolerance': 1e-4,
            'reference_source_sha256': sha(source)}


def audit_record(data, entry, annotation_index, pilot_output, reference_source):
    match = re.fullmatch(r'(?P<directory>\d+)/a_(?P<subject>\d+)_(?P<activity>\d+)_(?P<repeat>\d+)\.dat', entry.filename)
    if not match:
        raise ValueError('Unrecognized member name; do not guess metadata')
    subject, activity, repeat = [int(match[k]) for k in ('subject', 'activity', 'repeat')]
    if int(match['directory']) != subject:
        raise ValueError('Directory and nominal filename subject disagree')
    if len(data) % 575:
        raise ValueError('Nonuniform/truncated Intel frame size; needs separate decoder review')
    records = np.frombuffer(data, dtype=np.uint8).reshape(-1, 575)
    valid = ((records[:, 0] == 2) & (records[:, 1] == 61) & (records[:, 2] == 187)
             & (records[:, 11] == 3) & (records[:, 12] == 3)
             & (records[:, 19] == 40) & (records[:, 20] == 2))
    if not valid.all():
        raise ValueError(f'{int((~valid).sum())} frames differ from verified 3x3x30 layout; no silent skipping')
    timestamps, counter, permutation = fields(records)
    if not np.all(np.sort(permutation, axis=1) == np.array([0, 1, 2])):
        raise ValueError('Invalid antenna permutation')
    raw_delta = np.diff(timestamps)
    delta = np.where(raw_delta < -(2**31), raw_delta + 2**32, raw_delta)
    elapsed = np.concatenate([[0], np.cumsum(delta)]).astype(np.float64) / 1e6
    counter_delta = np.diff(counter) % (2**16)
    key = (subject, activity, repeat)
    annotation = annotation_index.get(key)
    category = ('single_action' if activity in range(7) else
                'annotated_continuous_candidate' if activity in range(10, 16) else 'unknown_activity_excluded')
    label = (1 if activity in (0, 1, 2) else 0) if activity in range(7) else None
    canonical = hashlib.sha256()
    canonical.update(f'Intel5300:antenna-corrected-int8-IQ:[{len(records)},3,3,30,2]:C'.encode())
    finite, total, zeros, rssi_zero = 0, 0, 0, 0
    amplitude_min, amplitude_max = math.inf, -math.inf
    pilot = entry.filename == PILOT
    matrix = np.empty((len(records), 90), dtype=np.float32) if pilot else None
    for start in range(0, len(records), 2048):
        stop = min(start + 2048, len(records))
        iq = canonical_iq(records[start:stop], permutation[start:stop])
        canonical.update(iq.tobytes(order='C'))
        scaled, undefined, no_rssi = rf_scaled_amplitude(iq, records[start:stop])
        mask = np.isfinite(scaled)
        finite += int(mask.sum()); total += scaled.size
        zeros += undefined; rssi_zero += no_rssi
        if mask.any():
            amplitude_min = min(amplitude_min, float(scaled[mask].min()))
            amplitude_max = max(amplitude_max, float(scaled[mask].max()))
        if pilot:
            matrix[start:stop] = scaled[:, 0].reshape(stop - start, 90)
    if pilot:
        np.save(pilot_output / 'pilot_native_tx0_amplitude_90.npy', matrix, allow_pickle=False)
        np.save(pilot_output / 'pilot_packet_elapsed_seconds.npy', elapsed, allow_pickle=False)
        ref = reference_check(reference_source, records, permutation)
        write_json(pilot_output / 'pilot_reference_check.json', ref)
    else:
        ref = None
    result = {
        'archive_member': entry.filename, 'source_kind': category,
        'nominal_subject_id': subject, 'activity_id': activity, 'repeat_id': repeat,
        'identifier_basis': 'filename/directory tokens cross-checked with annotation duration; preprocessing naming function unavailable',
        'single_action_binary_label': label, 'single_action_label_basis': 'official training activity-ID mapping' if label is not None else None,
        'bytes': len(data), 'zip_crc32_verified': True, 'raw_file_sha256': hashlib.sha256(data).hexdigest(),
        'canonical_csi_sha256': canonical.hexdigest(),
        'canonical_content_representation': 'header includes shape, then int8 IQ [packet,Tx,Rx-corrected,SC,IQ], C order; timestamps/RSSI excluded',
        'packet_count': len(records), 'packet_csi_shape': [3, 3, 30], 'raw_complex_dtype': 'complex64',
        'timestamp_span_seconds': float(elapsed[-1]), 'median_interval_microseconds': float(np.median(delta)),
        'p99_interval_microseconds': float(np.percentile(delta, 99)), 'max_interval_microseconds': int(delta.max()),
        'intervals_over_10ms': int((delta > 10000).sum()), 'intervals_over_100ms': int((delta > 100000).sum()),
        'nonincreasing_timestamp_intervals': int((delta <= 0).sum()),
        'timestamp_32bit_rollovers': int((raw_delta < -(2**31)).sum()),
        'bfee_counter_wraps': int((np.diff(counter) < 0).sum()),
        'bfee_counter_missing_measurements_if_forward_step': int((counter_delta[(counter_delta > 1) & (counter_delta < 32768)] - 1).sum()),
        'bfee_counter_duplicate_steps': int((counter_delta == 0).sum()),
        'bfee_counter_large_or_reset_steps': int((counter_delta >= 32768).sum()),
        'mean_observed_packet_rate_hz': float((len(records) - 1) / elapsed[-1]),
        'end_packet_index_time_minus_timestamp_seconds_at_assumed_1000hz': float((len(records) - 1) / 1000 - elapsed[-1]),
        'raw_noise_code_counts': {str(int(v)): int(n) for v, n in zip(*np.unique(records[:, 16], return_counts=True))},
        'antenna_selection_code_counts': {str(int(v)): int(n) for v, n in zip(*np.unique(records[:, 18], return_counts=True))},
        'rf_scaled_amplitude_finite_ratio': finite / total, 'rf_scaled_amplitude_values_checked': total,
        'rf_scaled_amplitude_range': [amplitude_min, amplitude_max],
        'zero_power_rx_sc_positions': zeros, 'packets_without_positive_rssi_sum': rssi_zero,
        'annotation_available': annotation is not None,
        'annotation_join_status': 'structural key matched; event time origin/semantics not independently verified' if annotation else 'no time annotation matched',
        'annotation_record': annotation, 'errors': [], 'reference_check': ref,
    }
    if annotation:
        fall = annotation['recorded_fall_time_seconds']
        duration = annotation['recorded_duration_seconds']
        nominal_index = fall * 1000
        result.update({'raw_duration_minus_annotation_seconds': float(elapsed[-1] - duration),
                       'nominal_fall_time_inside_raw_duration': bool(0 <= fall <= elapsed[-1]),
                       'fall_time_at_integer_packet_index_seconds': float(elapsed[nominal_index]) if nominal_index < len(elapsed) else None,
                       'packet_index_event_alignment_error_seconds': float(elapsed[nominal_index] - fall) if nominal_index < len(elapsed) else None})
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw-zip', type=Path, default=Path('/Users/chenxiaohe/Downloads/1.zip'))
    ap.add_argument('--code-zip', type=Path, default=Path('/Users/chenxiaohe/Downloads/TCS-Fall-main.zip'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--pilot-output', type=Path, required=True)
    args = ap.parse_args()
    if (args.output / 'audit.json').exists():
        raise FileExistsError('Use a new report directory to preserve earlier audits')
    args.output.mkdir(parents=True, exist_ok=True); args.pilot_output.mkdir(parents=True, exist_ok=True)
    old_hashes = json.loads(READ_ONLY_PROTECTED.read_text())
    if any(sha(MODULE / name) != digest for name, digest in old_hashes.items()):
        raise ValueError('Protected earlier artifact was already changed; inspect before continuing')
    inventories = [archive_inventory(p) for p in (args.raw_zip, args.code_zip)]
    write_json(args.output / 'archive_inventory.json', inventories)
    with zipfile.ZipFile(args.code_zip) as code:
        workbook_bytes = code.read('TCS-Fall-main/actual fall time.xlsx')
        if hashlib.sha256(workbook_bytes).hexdigest() != ANNOTATION_SHA:
            raise ValueError('Downloaded annotation workbook differs; audit its fields before joining')
        extracted_annotations = json.loads((MODULE / 'reports/tcs_fall_audit_v1/annotations_extracted.json').read_text())
        annotation_index = {(r['subject_id'], r['activity_id'], r['repeat_index_in_cell_zero_based']): r
                            for r in extracted_annotations}
        source_path = MODULE / 'outputs/tcs_fall_local_v1/source_package/TCS-Fall-main/code/wifilib.py'
        if sha(source_path) != hashlib.sha256(code.read('TCS-Fall-main/code/wifilib.py')).hexdigest():
            raise ValueError('Reference source mismatch')
        code_crc_error = code.testzip()
        if code_crc_error:
            raise ValueError('Code ZIP CRC failure: ' + code_crc_error)
        preprocess_present = any(PurePosixPath(n).name == 'preprocess.ipynb' for n in code.namelist())
    rows = []
    start_time = time.perf_counter()
    with zipfile.ZipFile(args.raw_zip) as raw:
        entries = raw.infolist()
        # The pilot is inspected first, then every remaining actual file, without extracting 2.4 GB.
        entries.sort(key=lambda e: (e.filename != PILOT, e.filename))
        for index, entry in enumerate(entries, 1):
            if entry.is_dir():
                continue
            try:
                data = raw.read(entry)  # ZipFile checks uncompressed size/CRC while reading.
                row = audit_record(data, entry, annotation_index, args.pilot_output, source_path)
                del data
            except Exception as error:
                row = {'archive_member': entry.filename, 'errors': [f'{type(error).__name__}: {error}']}
            rows.append(row)
            if index % 20 == 0 or index == 1 or '10' <= entry.filename.split('_')[2] <= '17':
                print(json.dumps({'audited': index, 'total': len(entries), 'file': entry.filename,
                                  'errors': row['errors'], 'elapsed_seconds': round(time.perf_counter() - start_time, 1)}), flush=True)
                write_json(args.output / 'progress.json', {'audited': index, 'total': len(entries)})
    write_json(args.output / 'records.json', rows)
    good = [r for r in rows if not r['errors']]
    annotated = [r for r in good if r['annotation_available']]
    subjects = sorted({r['nominal_subject_id'] for r in good})
    raw_groups, content_groups = defaultdict(list), defaultdict(list)
    for r in good:
        raw_groups[r['raw_file_sha256']].append(r['archive_member'])
        content_groups[r['canonical_csi_sha256']].append(r['archive_member'])
    after = {name: sha(MODULE / name) for name in old_hashes}
    if old_hashes != after:
        raise ValueError('Protected artifact changed during audit')
    archive_unchanged = all(p.stat().st_size == inv['bytes'] and p.stat().st_mtime_ns == inv['mtime_ns']
                            for p, inv in zip((args.raw_zip, args.code_zip), inventories))
    if not archive_unchanged:
        raise ValueError('Source archive changed during scan')
    summary = {
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'client_date': '2026-10-28',
        'source_archives': [{k: v for k, v in inv.items() if k != 'entries'} for inv in inventories],
        'archive_crc_check': {
            'raw_archive': {'entries_attempted': len(rows), 'entries_passed_full_record_audit': len(good),
                            'entries_failed': len(rows) - len(good), 'clean_archive': len(rows) == len(good)},
            'code_archive': 'all entries passed testzip',
        },
        'source_archives_size_and_mtime_unchanged': archive_unchanged,
        'all_raw_entries_audited': len(rows), 'records_with_errors': [r for r in rows if r['errors']],
        'subjects_from_available_archive': subjects, 'subject_identity_independently_verified': False,
        'activity_counts': dict(sorted(Counter(r['activity_id'] for r in good).items())),
        'source_kind_counts': dict(Counter(r['source_kind'] for r in good)),
        'single_action_class_counts_from_official_activity_mapping': dict(Counter(r['single_action_binary_label'] for r in good if r['source_kind'] == 'single_action')),
        'annotated_continuous_records_structurally_matched': len(annotated),
        'nominal_annotated_continuous_raw_hours': sum(r['timestamp_span_seconds'] for r in annotated) / 3600,
        'annotation_stated_hours_for_these_records': sum(r['annotation_record']['recorded_duration_seconds'] for r in annotated) / 3600,
        'nominal_event_time_outside_raw_count': sum(not r['nominal_fall_time_inside_raw_duration'] for r in annotated),
        'all_nominal_event_times_reference_raw_start_verified': False,
        'raw_annotation_duration_differences_seconds': [r['raw_duration_minus_annotation_seconds'] for r in annotated],
        'decoded_and_rf_scaled_values_checked': sum(r['rf_scaled_amplitude_values_checked'] for r in good),
        'recordings_with_rf_scaling_nonfinite': [r['archive_member'] for r in good if r['rf_scaled_amplitude_finite_ratio'] != 1],
        'recordings_with_nonincreasing_time': [r['archive_member'] for r in good if r['nonincreasing_timestamp_intervals']],
        'max_observed_packet_gap_ms': max(r['max_interval_microseconds'] for r in good) / 1000,
        'timestamp_rollover_count': sum(r['timestamp_32bit_rollovers'] for r in good),
        'exact_raw_duplicate_groups': [v for v in raw_groups.values() if len(v) > 1],
        'canonical_iq_duplicate_groups': [v for v in content_groups.values() if len(v) > 1],
        'preprocess_notebook_in_downloaded_code_archive': preprocess_present,
        'old_protected_artifacts_checked': len(old_hashes), 'old_artifacts_changed': [],
        'model_training_run': False, 'threshold_changed': False, 'test_evaluated': False,
        'event_recall': None, 'false_alarms_per_hour': None, 'alert_latency': None,
        'evaluation_limitations': [
            'Only one nominal subject archive is available; no independent cross-subject training/validation/test split is possible.',
            'Activity 16/17 has no inspected official semantics or time annotation; excluded from labels and exposure-time denominators.',
            'Structural annotation joins are supported by filenames and duration comparisons, not independently synchronized onset/impact ground truth.',
            'Missing preprocess.ipynb prevents claiming exact source preprocessing reproduction.',
            'Intel5300 native Tx0 features are not physically aligned with CSI-Bench HP232; HP weights/statistics are not applied.',
            'No gaps filled, time shifted, event labels inferred from amplitude, or random recording/window split created.',
        ],
    }
    write_json(args.output / 'audit.json', summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if summary['records_with_errors']:
        sys.exit(2)


if __name__ == '__main__':
    main()
