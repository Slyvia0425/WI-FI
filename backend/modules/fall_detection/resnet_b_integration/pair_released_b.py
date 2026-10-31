"""Run frozen B on exactly the raw HP validation segments already scored by C.

No training, threshold selection, new C test evaluation or HAR accuracy estimate.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from backend.shared.export import read_replay, write_replay
from backend.modules.fall_detection.activity_reference import segment_identity, attach_activity_reference
from backend.modules.human_activity_recognition.integration_v1.runtime import ReleasedActivityModel
from backend.modules.human_activity_recognition.preprocessing.hdf5_reader import read_csi_array
from backend.modules.human_activity_recognition.contract import validate_frame_values

MODULE = Path(__file__).resolve().parents[1]
B_MODULE = MODULE.parent / 'human_activity_recognition'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle', action='store_true')
    args = ap.parse_args()
    c = read_replay('fall_detection')
    if c.get('mode') != 'independent_segment_validation' or len(c['frames']) != 926:
        raise ValueError('Expected the existing 926 HP validation C outputs')
    original_scores = [(f['input_id'], f['fall_score'], f['threshold'], f['state']) for f in c['frames']]
    inputs = json.loads((MODULE / 'reports/resnet_b_integration_v1/inputs_manifest.json').read_text())
    by_id = {r['sample_id']: r for r in inputs}
    b = ReleasedActivityModel()
    frames = []
    start = perf_counter()
    for offset in range(0, len(c['frames']), 32):
        group = c['frames'][offset:offset + 32]
        arrays = []
        for frame in group:
            row = by_id[frame['sample_id']]
            a = read_csi_array(Path(row['file_path']), ('CSI_amps',))
            if a.shape != (232, 500, 1):
                raise ValueError('Unexpected raw HP HDF5 shape')
            a = a[:, :, 0]
            if segment_identity(a) != frame['input_id']:
                raise ValueError('Raw B/C input identity mismatch')
            arrays.append(a)
        for frame, prediction in zip(group, b.predict_many(arrays)):
            frames.append({**prediction, 't': frame['t'], 'sample_index': frame['sample_index'],
                'sample_id': frame['sample_id'], 'input_id': frame['input_id'], 'device': 'HP',
                'recording_id': frame['recording_id'], 'time_basis': 'independent segment index, not seconds',
                'activity_ground_truth_available': False})
    elapsed = perf_counter() - start
    cfg = b.config
    payload = {
        'module': 'human_activity_recognition', 'version': 'released-b-paired-hp-v1', 'mock': False,
        'model_version': cfg['model_version'], 'dataset': 'CSI-Bench / FallDetection (B inference only)',
        'split': 'C validation inputs; no five-class HAR ground truth', 'mode': 'paired_independent_segments',
        'timeline_kind': 'presentation_only', 'physical_timestamps_available': False,
        'provenance': {'kind': 'real_released_B_inference_on_same_C_input', 'trained_artifact_available': True,
            'source_commit': cfg['source_commit'], 'release_url': cfg['release_url'],
            'checkpoint_sha256': cfg['checkpoint_sha256'], 'joint_ground_truth': False,
            'context_alignment': 'exact raw amplitude input_id; no display-time join',
            'inference_reproduced_locally': True, 'new_training': False},
        'frames': frames,
        'preprocessing': {'tiles': [{'label': 'Input', 'value': 'Same 926 HP raw amplitude segments as C'},
            {'label': 'Features', 'value': '4195 spectral / temporal → fixed 2800'},
            {'label': 'Normalization', 'value': 'Spectral branch raw amplitude, no CNN z-score'},
            {'label': 'Ground truth', 'value': 'B activity labels unavailable on FallDetection'}]},
        'training': {'table': [{'model': 'Author-released spectral ExtraTrees', 'seed': 42,
            'trees': 900, 'integration_training': 'None; frozen author parameters'}]},
        'experiments': {'table': [{'scope': 'B/C integration', 'measure': 'Same-input predictions', 'value': len(frames)},
            {'scope': 'B/C integration', 'measure': 'Fall class in B', 'value': 'Absent; B context cannot confirm/veto C'}]},
        'metrics': {'table': [{'scope': 'Current paired inputs', 'measure': 'B predictions', 'value': len(frames)},
            {'scope': 'Current paired inputs', 'measure': 'B activity accuracy / joint accuracy', 'value': 'Not measurable without activity annotations'},
            {'scope': 'Current paired inputs', 'measure': 'Confidence >=0.6', 'value': sum(f['stable'] for f in frames)}]},
        'notice': 'Real frozen B inference. These independent C validation segments have no B activity labels. A five-class prediction on Fall is a model preference, not a verified action.',
    }
    problems = validate_frame_values(payload)
    if problems:
        raise ValueError(problems)
    combined = attach_activity_reference(c, payload)
    combined['provenance']['paired_module_B'] = True
    combined['provenance']['B_pairing'] = 'same raw HP input, verified by canonical content identity'
    for row in combined['metrics']['table']:
        if row.get('measure') == 'Joint accuracy':
            row['value'] = 'Not evaluated; paired activity ground truth unavailable'
    assert original_scores == [(f['input_id'], f['fall_score'], f['threshold'], f['state']) for f in combined['frames']]
    summary = {'records': len(frames), 'reference': combined['activity_reference_summary'],
        'B_activity_counts': dict(Counter(f['activity'] for f in frames)),
        'confidence_ge_0_6': sum(f['stable'] for f in frames), 'B_export_wall_seconds': elapsed,
        'C_scores_thresholds_states_unchanged': True, 'HAR_activity_ground_truth_available': False,
        'joint_accuracy_evaluated': False, 'new_training': False, 'new_C_test_predictions': False,
        'B_checkpoint_sha256': cfg['checkpoint_sha256']}
    report = MODULE / 'reports/resnet_b_integration_v2'
    report.mkdir(parents=True, exist_ok=True)
    (report / 'paired_inference_receipt.json').write_text(json.dumps(summary, indent=2))
    # Historical fixture/replay is preserved before switching the default output.
    for name in ('human_activity_recognition', 'fall_detection'):
        output = MODULE.parent / name / 'outputs'
        saved = output / 'before_real_B_v1.json'
        if not saved.exists():
            saved.write_text(json.dumps(read_replay(name), indent=1))
    write_replay('human_activity_recognition', payload, bundle=args.bundle)
    write_replay('fall_detection', combined, bundle=args.bundle)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
