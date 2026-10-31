"""Independently verify saved TCS normalization without training or test inference.

The numerical reference merges float64 batch means and centered sums of squares
(M2), rather than reusing prepare_features.py's raw sums/squared-sums helper.
All model/data files are read-only. The output must be a new JSON report.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
MODULE = HERE.parent
sys.path.insert(0, str(HERE))
from dataset import GCVDataset


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def cnn_ast(path):
    classes = [node for node in ast.parse(path.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == 'CNN']
    if len(classes) != 1:
        raise ValueError(f'Expected exactly one CNN class: {path}')
    return ast.dump(classes[0], include_attributes=False)


def independent_moments(cache, indices, batch_size=16):
    """Merge centered batch moments; population variance uses denominator n."""
    n = 0
    mean = np.zeros(90, dtype=np.float64)
    m2 = np.zeros(90, dtype=np.float64)
    for start in range(0, len(indices), batch_size):
        x = np.asarray(cache[indices[start:start + batch_size]],
                       dtype=np.float64).reshape(-1, 90)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite selected training feature')
        batch_n = len(x)
        batch_mean = x.mean(axis=0)
        batch_m2 = np.square(x - batch_mean).sum(axis=0)
        delta = batch_mean - mean
        total = n + batch_n
        m2 += batch_m2 + delta * delta * n * batch_n / total
        mean += delta * batch_n / total
        n = total
    if n == 0:
        raise ValueError('No training observations')
    return n, mean, np.sqrt(m2 / n)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--features', type=Path,
                    default=MODULE / 'outputs/tcs_cross_subject_v1/features_v1')
    ap.add_argument('--checkpoint', type=Path,
                    default=MODULE / 'outputs/tcs_cross_subject_v1/comparison_seed42_v1/SA_CT_standardized_GCV/best.pt')
    ap.add_argument('--errors', type=Path,
                    default=MODULE / 'reports/tcs_cross_subject_v1/comparison_seed42_v1/evaluation/D_single_action_errors.json')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing diagnostic report; use a fresh output')
    if not args.output.parent.is_dir():
        raise FileNotFoundError('Create the designated report directory first')

    prep = json.loads((args.features / 'preparation.json').read_text())
    rows = json.loads((args.features / 'samples.json').read_text())
    if {r['subject'] for r in rows} != {1, 2, 3}:
        raise ValueError('Unexpected subjects, or test features included')
    if tuple(prep['shape'][1:]) != (1, 400, 90) or prep['shape'][0] != len(rows):
        raise ValueError('Unexpected feature axes or sample count')
    cache_hash = sha(args.features / 'features.f32')
    if cache_hash != prep['feature_cache_sha256']:
        raise ValueError('Feature cache changed')
    if sha(args.features / 'samples.json') != prep['sample_manifest_sha256']:
        raise ValueError('Sample manifest changed')
    cache = np.memmap(args.features / 'features.f32', dtype=np.float32,
                      mode='r', shape=tuple(prep['shape']))

    summaries = []
    normalizers = {}
    for include_ct, name in ((False, 'normalization_sa.json'),
                             (True, 'normalization_sa_ct.json')):
        saved = json.loads((args.features / name).read_text())
        normalizers[name] = saved
        indices = [i for i, r in enumerate(rows)
                   if r['role'] == 'train'
                   and (include_ct or r['kind'] == 'single_action')]
        subjects = sorted({rows[i]['subject'] for i in indices})
        if subjects != [1, 2] or saved['fit_subjects'] != subjects or saved['fit_role'] != 'train':
            raise ValueError('Normalizer contains non-training subjects')
        if saved['feature_cache_sha256'] != cache_hash:
            raise ValueError('Normalizer feature provenance mismatch')
        if saved['includes_continuous_negatives'] != include_ct:
            raise ValueError('Normalizer selected the wrong training pool')
        n, mean, std = independent_moments(cache, indices)
        if saved['fit_sample_count'] != len(indices) or saved['time_points_per_feature'] != n:
            raise ValueError('Normalizer training count mismatch')
        # Independent algorithm changes summation order; allow round-off only.
        np.testing.assert_allclose(mean, saved['mean'], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(std, saved['std'], atol=1e-12, rtol=1e-12)
        constant = np.flatnonzero(std < saved['small_std_threshold']).tolist()
        if constant != saved['constant_feature_indices']:
            raise ValueError('Constant-feature detection mismatch')
        expected_scale = np.where(std < saved['small_std_threshold'], 1., std)
        np.testing.assert_allclose(expected_scale, saved['scale'], atol=1e-12, rtol=1e-12)
        if min(saved['scale']) <= 0:
            raise ValueError('Nonpositive normalization scale')
        summaries.append({
            'name': name, 'algorithm': 'float64 batch mean/M2 merge; population variance',
            'train_subjects': subjects, 'fit_samples': len(indices), 'feature_timepoints': n,
            'max_abs_mean_error': float(np.max(np.abs(mean - saved['mean']))),
            'max_abs_std_error': float(np.max(np.abs(std - saved['std']))),
            'scale_min': min(saved['scale']), 'scale_max': max(saved['scale']),
            'recomputed_constant_feature_count': len(constant),
            'saved_constant_feature_count': len(saved['constant_feature_indices']),
            'absolute_tolerance': 1e-12, 'relative_tolerance': 1e-12,
            'passed': True,
        })

    d_norm = normalizers['normalization_sa_ct.json']
    c_norm = normalizers['normalization_sa.json']
    checkpoint_hash = sha(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['normalization_parameters'] != d_norm:
        raise ValueError('D checkpoint normalizer differs from prepared D parameters')
    if checkpoint['label_mapping'] != {'Nonfall': 0, 'Fall': 1}:
        raise ValueError('Unexpected checkpoint label mapping')
    if checkpoint['data_version']['feature_cache_sha256'] != cache_hash:
        raise ValueError('Checkpoint and feature cache provenance differ')

    model_source = MODULE / 'tcs_withinsubject_v1/model.py'
    official_source = MODULE / 'outputs/tcs_fall_audit_v1/upstream/code/models.py'
    if cnn_ast(model_source) != cnn_ast(official_source):
        raise ValueError('CNN implementation differs from fixed official source')
    if sha(model_source) != checkpoint['run_config']['model_source_sha256']:
        raise ValueError('Model source differs from checkpoint provenance')

    errors = json.loads(args.errors.read_text())
    targets = {r['sample_id'] for r in errors if r['kind'] == 'FN'}
    if len(targets) != 6:
        raise ValueError('Expected the six already-reported D false negatives')
    ds = GCVDataset(args.features, 'validation', True,
                    args.features / 'normalization_sa_ct.json')
    checks = []
    for item, index in enumerate(ds.indices):
        if rows[index]['sample_id'] not in targets:
            continue
        x, y, sample_id = ds[item]
        expected = (np.array(cache[index], dtype=np.float32)
                    - np.asarray(d_norm['mean'], dtype=np.float32)) / np.asarray(d_norm['scale'], dtype=np.float32)
        if not np.array_equal(x.numpy(), expected):
            raise ValueError('Dataset differs from direct float32 normalization')
        if tuple(x.shape) != (1, 400, 90) or int(y) != 1 or rows[index]['subject'] != 3:
            raise ValueError('Unexpected FN tensor, label, or subject')
        if not np.isfinite(x.numpy()).all():
            raise ValueError('Nonfinite normalized FN input')
        checks.append({
            'sample_id': sample_id, 'subject': rows[index]['subject'],
            'activity': rows[index]['activity'], 'shape': list(x.shape), 'label': int(y),
            'max_abs_difference_dataset_vs_manual': float(np.max(np.abs(x.numpy() - expected))),
            'finite': True, 'range': [float(x.min()), float(x.max())], 'passed': True,
        })
    if len(checks) != 6:
        raise ValueError('Not all six false negatives were independently checked')
    if sha(args.checkpoint) != checkpoint_hash:
        raise ValueError('Checkpoint changed during read-only checks')
    ratio = np.asarray(d_norm['scale']) / np.asarray(c_norm['scale'])
    result = {
        'status': 'passed', 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'independent statistics and input implementation checks, not new performance estimates',
        'training_run': False, 'test_subject_read': False, 'model_predictions_run': False,
        'independent_normalization': summaries,
        'D_over_C_scale_min_median_max': np.quantile(ratio, [0, .5, 1]).tolist(),
        'D_checkpoint_normalizer_exact_match': True,
        'D_FN_dataset_checks': checks,
        'official_CNN_class_AST_equal': True,
        'checkpoint_unchanged': True,
        'source_hashes': {
            'features': cache_hash, 'samples': sha(args.features / 'samples.json'),
            'normalization_C': sha(args.features / 'normalization_sa.json'),
            'normalization_D': sha(args.features / 'normalization_sa_ct.json'),
            'checkpoint_D': checkpoint_hash, 'CNN': sha(model_source),
            'official_models': sha(official_source), 'six_FN_source': sha(args.errors),
            'Dataset': sha(HERE / 'dataset.py'), 'diagnostic_script': sha(Path(__file__)),
        },
    }
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
