"""Test-only fan on/off slice evaluation of the unchanged breathing LSTM."""
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ..inference.frozen_detector import FrozenBreathingDetector
from ..preprocessing.data import BreathingDataset
from ..preprocessing.temporal import extract_interpolated_subcarriers
from .evaluate_spike_preprocessing import ROOT, digest

SCENARIOS = ('LOS_FanOff', 'LOS_FanOn', 'NLOS_FanOff', 'NLOS_FanOn')
CHECKPOINT_SHA = 'fff3cdb28c7577489400be7cc82506f074703ea22425404be1cc3224923df956'
THRESHOLD = 0.054542455822229385


def detection_metrics(rows, threshold):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Expected threshold in [0,1]')
    total_positive = sum(r['label'] == 1 for r in rows)
    if any(r['label'] not in (0, 1) for r in rows):
        raise ValueError('Unknown binary label')
    kept = [r for r in rows if r['probability'] is not None]
    if any(not math.isfinite(r['probability']) or not 0 <= r['probability'] <= 1 for r in kept):
        raise ValueError('Nonfinite or invalid probability')
    tp = sum(r['label'] == 1 and r['probability'] >= threshold for r in kept)
    fn = sum(r['label'] == 1 and r['probability'] < threshold for r in kept)
    fp = sum(r['label'] == 0 and r['probability'] >= threshold for r in kept)
    tn = sum(r['label'] == 0 and r['probability'] < threshold for r in kept)
    ratio = lambda numerator, denominator: numerator/denominator if denominator else None
    macro_f1 = None
    if tp+fn and tn+fp:
        macro_f1 = (2*tp/(2*tp+fp+fn)+2*tn/(2*tn+fp+fn))/2
    probability_summary = {}
    for label, name in ((0, 'empty'), (1, 'sleep')):
        values = [r['probability'] for r in kept if r['label'] == label]
        probability_summary[name] = {'count': len(values), 'min_p10_median_p90_max':
            np.quantile(values, [0, .1, .5, .9, 1]).tolist() if values else None}
    return {'metadata_samples': len(rows), 'available': len(kept), 'unavailable': len(rows)-len(kept),
        'coverage': ratio(len(kept), len(rows)),
        'metadata_classes': {'sleep': total_positive, 'empty': len(rows)-total_positive},
        'available_classes': {'sleep': tp+fn, 'empty': tn+fp},
        'confusion_matrix': {'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp},
        'breathing_recall': ratio(tp, tp+fn), 'empty_false_positive_rate': ratio(fp, tn+fp),
        'specificity': ratio(tn, tn+fp), 'precision': ratio(tp, tp+fp),
        'accuracy': ratio(tp+tn, len(kept)), 'macro_f1': macro_f1,
        'positive_detection_fraction_of_all_labeled': ratio(tp, total_positive),
        'probability_summaries': probability_summary,
        'unavailable_reasons': dict(Counter(reason for r in rows if r['probability'] is None for reason in r['unavailable_reasons']))}


def on_off_difference(off, on):
    result = {}
    for key in ('breathing_recall', 'empty_false_positive_rate'):
        result[key+'_on_minus_off_percentage_points'] = 100*(on[key]-off[key]) if on[key] is not None and off[key] is not None else None
    return result


def report_slices(rows, threshold):
    devices = sorted({r['device_id'] for r in rows})
    scenarios = {scenario: detection_metrics([r for r in rows if r['scenario'] == scenario], threshold) for scenario in SCENARIOS}
    by_device = {scenario: {device: detection_metrics([r for r in rows if r['scenario'] == scenario and r['device_id'] == device], threshold)
                           for device in devices} for scenario in SCENARIOS}
    standardized = {}
    for scenario in SCENARIOS:
        standardized[scenario] = {}
        for metric in ('breathing_recall', 'empty_false_positive_rate'):
            available = {device: m[metric] for device, m in by_device[scenario].items() if m[metric] is not None}
            standardized[scenario][metric] = float(np.mean(list(available.values()))) if available else None
            standardized[scenario][metric+'_devices'] = sorted(available)
    comparisons = {}
    for geometry in ('LOS', 'NLOS'):
        off, on = geometry+'_FanOff', geometry+'_FanOn'
        comparisons[geometry] = {'pooled': on_off_difference(scenarios[off], scenarios[on]),
            'by_device': {device: on_off_difference(by_device[off][device], by_device[on][device]) for device in devices},
            'equal_device_matched': {}}
        for metric in ('breathing_recall', 'empty_false_positive_rate'):
            matched = [device for device in devices if by_device[off][device][metric] is not None and by_device[on][device][metric] is not None]
            comparisons[geometry]['equal_device_matched'][metric] = {
                'devices': matched, 'on_minus_off_percentage_points':
                float(np.mean([100*(by_device[on][device][metric]-by_device[off][device][metric]) for device in matched])) if matched else None}
    return {'overall_fan_subset': detection_metrics(rows, threshold), 'by_scenario': scenarios,
            'by_scenario_device': by_device, 'equal_device_scenario_means': standardized, 'on_off_comparisons': comparisons}


@torch.no_grad()
def infer_batch(detector, sequences, indices, rows):
    if not sequences:
        return
    tensor = torch.from_numpy(np.stack(sequences).astype(np.float32, copy=False))
    probabilities = torch.sigmoid(detector.model(tensor)).cpu().numpy()
    if not np.isfinite(probabilities).all():
        raise ValueError('Model produced nonfinite probabilities')
    for index, p in zip(indices, probabilities):
        rows[index]['probability'] = float(p)
        rows[index]['prediction'] = int(p >= detector.threshold)


def main():
    output = ROOT/'fan_detection_v1'
    if output.exists():
        raise FileExistsError('Preserve completed frozen fan evaluation')
    checkpoint = ROOT/'lstm_interpolated_full_seed123/best_model.pt'
    protected = json.loads((ROOT/'phase_feasibility_v1/config.json').read_text())['protected_artifact_hashes']
    if digest(checkpoint) != CHECKPOINT_SHA or any(digest(path) != sha for path, sha in protected.items()):
        raise ValueError('Frozen checkpoint/application artifact changed')
    source = BreathingDataset('data/BreathingDetection', 'test_id', normalize=True)
    selected_indices = [i for i, row in enumerate(source._rows) if (row.get('deplyoment') or row.get('deployment')) in SCENARIOS]
    selected_metadata = [source._rows[i] for i in selected_indices]
    if len(selected_indices) != 4156 or len({r['id'] for r in selected_metadata}) != 4156:
        raise ValueError('Fan test subset changed; review protocol instead of silently changing it')
    base = Path('data/BreathingDetection')
    split_ids = {name: set(json.loads((base/'splits'/f'{name}.json').read_text())) for name in ('train_id', 'val_id', 'test_id')}
    ids = {r['id'] for r in selected_metadata}
    if ids & (split_ids['train_id'] | split_ids['val_id']) or not ids <= split_ids['test_id']:
        raise ValueError('Train/validation membership in scored fan subset')
    metadata_hashes = {str(base/'metadata/sample_metadata.csv'): digest(base/'metadata/sample_metadata.csv'),
                       **{str(p): digest(p) for p in (base/'splits').glob('*.json')}}
    raw_hashes = {str(base/r['file_path']): digest(base/r['file_path']) if (base/r['file_path']).is_file() else None for r in selected_metadata}
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    detector = FrozenBreathingDetector(checkpoint)
    if detector.threshold != THRESHOLD:
        raise ValueError('Frozen threshold mismatch')
    config = {'checkpoint_sha256': CHECKPOINT_SHA, 'threshold': THRESHOLD, 'device': 'cpu', 'torch_threads': 4,
        'batch_size': 128, 'scenarios': list(SCENARIOS), 'split': 'test_id', 'selected_count': len(selected_indices),
        'protocol_document_sha256': digest(Path(__file__).parents[1]/'FAN_DETECTION_PROTOCOL.md'),
        'evaluation_sha256': digest(__file__),
        'loader_sha256': digest(Path(__file__).parents[1]/'preprocessing/data.py'),
        'feature_sha256': digest(Path(__file__).parents[1]/'preprocessing/temporal.py'),
        'metadata_sha256': metadata_hashes, 'raw_file_sha256': raw_hashes,
        'protected_artifact_hashes': protected, 'application_enabled': False,
        'limitations': ['Previously observed official test, not new independent validation.',
            'Dataset sleep/empty proxies, no respiratory waveform ground truth or BPM evaluation.',
            'Source U23/E09 only, four devices; no proven independent/pairwise captures.',
            'LOS fan-on empty class is missing; cannot estimate its FPR.',
            'Historic aggregate used MPS; this deterministic CPU inference is not a bitwise accelerator replay.']}
    output.mkdir()
    (output/'config.json').write_text(json.dumps(config, indent=2))
    rows, sequences, batch_indices = [], [], []
    for source_index in selected_indices:
        metadata = source._rows[source_index]
        record = {'sample_id': metadata['id'], 'split': 'test_id', 'label': {'empty': 0, 'sleep': 1}[metadata['label']],
            'label_name': metadata['label'], 'scenario': metadata.get('deplyoment') or metadata.get('deployment'),
            'device_id': metadata['device'], 'participant_id': metadata['user'], 'environment_id': metadata['environment'],
            'difficulty': metadata['difficulty'], 'metadata_num_subcarriers': int(metadata['num_sub']),
            'probability': None, 'prediction': None, 'unavailable_reasons': []}
        rows.append(record)
        try:
            sample = source[source_index]
            record['original_num_subcarriers'] = int(sample.x.shape[1])
            if sample.assessable:
                sequence = extract_interpolated_subcarriers(sample.x)
                if sequence.shape != (300, 232):
                    raise ValueError('Frozen temporal layout mismatch')
                sequences.append(sequence)
                batch_indices.append(len(rows)-1)
            else:
                record['unavailable_reasons'] = list(sample.quality_flags)
        except (OSError, ValueError) as error:
            record['unavailable_reasons'] = [f'{type(error).__name__}: {error}']
        if len(sequences) >= 128:
            infer_batch(detector, sequences, batch_indices, rows)
            sequences, batch_indices = [], []
        if len(rows) % 512 == 0:
            print('Processed', len(rows), '/', len(selected_indices), 'fan test samples', flush=True)
    infer_batch(detector, sequences, batch_indices, rows)
    report = report_slices(rows, detector.threshold)
    report['comparability'] = {'participants': sorted({r['participant_id'] for r in rows}),
        'environments': sorted({r['environment_id'] for r in rows}), 'devices': sorted({r['device_id'] for r in rows}),
        'scenario_difficulty_counts': {s: dict(Counter(r['difficulty'] for r in rows if r['scenario'] == s)) for s in SCENARIOS},
        'scenario_carrier_counts': {s: dict(Counter(str(r.get('original_num_subcarriers')) for r in rows if r['scenario'] == s)) for s in SCENARIOS},
        'selected_raw_unique_hashes': len({sha for sha in raw_hashes.values() if sha is not None}),
        'selected_exact_duplicate_file_groups': sum(count > 1 for sha, count in Counter(raw_hashes.values()).items() if sha is not None),
        'independence_warning': 'No exhaustive train/test signal-duplication audit or paired source recording verification; no independent-window CIs.'}
    report['application_enabled'] = False
    report['evaluation'] = 'frozen_fan_presence_test_subgroup_audit'
    errors = [r for r in rows if r['prediction'] is not None and r['prediction'] != r['label']]
    unavailable = [r for r in rows if r['probability'] is None]
    for path, sha in {**metadata_hashes, **protected}.items():
        if digest(path) != sha:
            raise ValueError('Protected metadata/application changed')
    for path, sha in raw_hashes.items():
        current = digest(path) if Path(path).is_file() else None
        if current != sha:
            raise ValueError('Selected raw HDF5 changed')
    (output/'predictions.json').write_text(json.dumps(rows, indent=2))
    (output/'metrics.json').write_text(json.dumps(report, indent=2))
    (output/'error_cases.json').write_text(json.dumps(errors, indent=2))
    (output/'unavailable.json').write_text(json.dumps(unavailable, indent=2))
    print(json.dumps({key: report[key] for key in ('by_scenario', 'on_off_comparisons', 'comparability')}, indent=2))


if __name__ == '__main__':
    main()
