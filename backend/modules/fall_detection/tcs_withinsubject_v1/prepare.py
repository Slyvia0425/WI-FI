"""Reconstruct a recorded, within-subject TCS-Fall pilot; never inspect continuous CSI."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from scipy import signal

MODULE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODULE / 'tcs_fall_audit_v1'))
from audit_local_archives import fields, canonical_iq, rf_scaled_amplitude

RECIPE = {
    'version': 'tcs_public_code_reconstruction_v1', 'input_packets': 8000,
    'native_packet_layout': ['Tx=3', 'Rx=3', 'reported_subcarrier_position=30'],
    'selected_tx_index': 0, 'selected_feature_order': 'Rx-major then subcarrier; 3x30=90',
    'amplitude': 'antenna-corrected RF-scaled complex64 abs',
    'filter': {'kind': 'Butterworth', 'order': 8, 'normalized_cutoff': 0.2,
               'application': 'scipy.signal.filtfilt on the available 8000-packet segment only',
               'physical_cutoff_hz': '100 only under ideal uniform 1000Hz; packet timestamps are irregular'},
    'gcv': {'group_size': 20, 'std_ddof': 0, 'formula': 'std/mean; exactly zero mean returns zero'},
    'output_layout': ['channel=1', 'group_time=400', 'native_features=90'],
    'output_dtype': 'float32', 'train_fitted_normalization': None,
    'padding': False, 'resampling': False, 'augmentation': False,
    'scope': 'offline segment feature reconstruction; not a faithful missing-preprocess.ipynb reproduction',
    'record_tail_policy': 'take first8000 packet positions as in public csi-view notebook; log omitted tail',
    'causality': 'within-segment bidirectional filtering; no claim of per-packet zero-delay streaming',
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def transform_dat_bytes(data):
    if len(data) % 575:
        raise ValueError('Unsupported/truncated packet layout')
    records = np.frombuffer(data, dtype=np.uint8).reshape(-1, 575)
    if len(records) < 8000:
        raise ValueError('Fewer than8000 packets; no implicit padding')
    if not np.all((records[:, 0] == 2) & (records[:, 1] == 61) & (records[:, 2] == 187)
                  & (records[:, 11] == 3) & (records[:, 12] == 3)
                  & (records[:, 19] == 40) & (records[:, 20] == 2)):
        raise ValueError('Packet layout differs from audited Intel5300 data')
    selected = records[:8000]
    timestamps, _, permutation = fields(selected)
    if not np.all(np.sort(permutation, axis=1) == np.array([0, 1, 2])):
        raise ValueError('Invalid Rx permutation')
    native = []
    for start in range(0, len(selected), 1024):
        block = selected[start:start + 1024]
        iq = canonical_iq(block, permutation[start:start + len(block)])
        amplitude, undefined, rssi_zero = rf_scaled_amplitude(iq, block)
        if undefined or rssi_zero or not np.isfinite(amplitude).all():
            raise ValueError('RF scaling failure; no NaN fill')
        native.append(amplitude[:, 0].reshape(len(block), 90))
    amplitude = np.concatenate(native)
    b, a = signal.butter(8, 0.2, 'lowpass')
    filtered = signal.filtfilt(b, a, amplitude, axis=0)
    grouped = filtered.reshape(400, 20, 90)
    means, stds = grouped.mean(axis=1), grouped.std(axis=1, ddof=0)
    gcv = np.divide(stds, means, out=np.zeros_like(stds), where=means != 0).astype(np.float32)
    if gcv.shape != (400, 90) or not np.isfinite(gcv).all():
        raise ValueError('Invalid GCV features')
    delta = np.diff(timestamps)
    delta = np.where(delta < -(2**31), delta + 2**32, delta)
    if (delta <= 0).any():
        raise ValueError('Nonincreasing native timestamps')
    return gcv[None], {
        'source_packets': len(records), 'selected_packets': 8000, 'omitted_tail_packets': len(records) - 8000,
        'selected_timestamp_span_seconds': float(delta.sum() / 1e6),
        'group_mean_exactly_zero_positions': int((means == 0).sum()),
        'negative_gcv_values': int((gcv < 0).sum()), 'gcv_min': float(gcv.min()), 'gcv_max': float(gcv.max()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive', type=Path, default=Path('/Users/chenxiaohe/Downloads/1.zip'))
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if (args.output / 'features.npz').exists():
        raise FileExistsError('Use a fresh experiment directory')
    args.output.mkdir(parents=True, exist_ok=True)
    audited = json.loads((MODULE / 'reports/tcs_fall_local_v1/records.json').read_text())
    receipt = json.loads((MODULE / 'reports/tcs_fall_local_v1/audit.json').read_text())
    if sha(args.archive) != receipt['source_archives'][0]['sha256']:
        raise ValueError('Archive differs from audited version; audit it first')
    selected = sorted((r for r in audited if not r['errors'] and r['source_kind'] == 'single_action'),
                      key=lambda r: r['archive_member'])
    if len(selected) != 207:
        raise ValueError('Expected actual audited207 recordings')
    generator = np.random.default_rng(42)
    validation_ids = set()
    for activity in range(7):
        group = sorted(r['archive_member'] for r in selected if r['activity_id'] == activity)
        if len(group) < 6:
            raise ValueError('Insufficient activity group')
        validation_ids.update(generator.permutation(group)[:6].tolist())
    # Freeze recording IDs before constructing features; no test split is invented.
    split = {
        'protocol': 'single nominal subject01; stratified by activity; six whole recordings per activity in validation',
        'seed': 42, 'train_ids': [r['archive_member'] for r in selected if r['archive_member'] not in validation_ids],
        'validation_ids': sorted(validation_ids),
        'corrupt_excluded': [r['archive_member'] for r in audited if r['errors']],
        'continuous_reserved_unread': [r['archive_member'] for r in audited if r.get('source_kind') == 'annotated_continuous_candidate'],
        'unknown_activity_excluded': [r['archive_member'] for r in audited if r.get('source_kind') == 'unknown_activity_excluded'],
        'user_override': 'User explicitly requested starting training before redownload; three known-corrupt records excluded and disclosed.',
        'nearby_repeat_and_session_independence': 'not established; no claim of cross-subject/session generalization',
    }
    if set(split['train_ids']) & set(split['validation_ids']):
        raise ValueError('Recording overlap')
    train_hashes = {r['canonical_csi_sha256'] for r in selected if r['archive_member'] in split['train_ids']}
    val_hashes = {r['canonical_csi_sha256'] for r in selected if r['archive_member'] in validation_ids}
    if train_hashes & val_hashes:
        raise ValueError('Identical canonical CSI across train and validation')
    (args.output / 'split.json').write_text(json.dumps(split, indent=2))
    x, y, is_validation, metadata = [], [], [], []
    with zipfile.ZipFile(args.archive) as archive:
        for index, row in enumerate(selected, 1):
            data = archive.read(row['archive_member'])
            if hashlib.sha256(data).hexdigest() != row['raw_file_sha256']:
                raise ValueError('Record content differs from audit')
            features, info = transform_dat_bytes(data)
            x.append(features); y.append(row['single_action_binary_label'])
            is_validation.append(row['archive_member'] in validation_ids)
            metadata.append({'id': row['archive_member'], 'label': row['single_action_binary_label'],
                             'activity_id': row['activity_id'], **info})
            if index % 30 == 0:
                print(json.dumps({'prepared': index, 'total': len(selected), 'shape': list(features.shape)}), flush=True)
    x = np.stack(x); y = np.array(y, dtype=np.int64); mask = np.array(is_validation)
    np.savez_compressed(args.output / 'features.npz', x=x, y=y, is_validation=mask,
                        ids=np.array([r['archive_member'] for r in selected]))
    summary = {
        'version': 'tcs_subject01_gcv_cnn_pilot_v1', 'recipe': RECIPE,
        'archive': str(args.archive), 'archive_sha256': receipt['source_archives'][0]['sha256'],
        'records': metadata, 'shape': list(x.shape), 'finite_ratio': float(np.isfinite(x).mean()),
        'gcv_range': [float(x.min()), float(x.max())],
        'train_count': int((~mask).sum()), 'validation_count': int(mask.sum()),
        'train_fall_nonfall': [int((y[~mask] == 1).sum()), int((y[~mask] == 0).sum())],
        'validation_fall_nonfall': [int((y[mask] == 1).sum()), int((y[mask] == 0).sum())],
        'selected_timestamp_span_range_seconds': [min(r['selected_timestamp_span_seconds'] for r in metadata), max(r['selected_timestamp_span_seconds'] for r in metadata)],
        'features_sha256': sha(args.output / 'features.npz'), 'split_sha256': sha(args.output / 'split.json'),
        'prepare_source_sha256': sha(__file__), 'native_decoder_sha256': sha(MODULE / 'tcs_fall_audit_v1/audit_local_archives.py'),
        'test_evaluated': False, 'continuous_records_read_for_this_preparation': 0,
    }
    (args.output / 'preparation.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in ('records', 'recipe')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
