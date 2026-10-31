"""Re-evaluate only the42 pilot validation recordings in a fresh CPU process."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from model import CNN
from prepare import sha
from train import metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(2)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if sha(args.data_dir / 'features.npz') != state['run_config']['features_sha256']:
        raise ValueError('Feature cache changed')
    source = Path(__file__).parent
    for name in ('model.py', 'prepare.py'):
        if sha(source / name) != state['run_config']['source_sha256'][name]:
            raise ValueError('Saved source changed')
    model = CNN().eval(); model.load_state_dict(state['model_state_dict'], strict=True)
    data = np.load(args.data_dir / 'features.npz', allow_pickle=False)
    mask = data['is_validation']; x = data['x'][mask]; y = data['y'][mask]; ids = data['ids'][mask]
    if len(y) != 42 or set(ids) != set(state['split_identifiers']['validation_ids']):
        raise ValueError('Validation split mismatch')
    scores = []
    with torch.inference_mode():
        for start in range(0, len(x), 2):
            result = model(torch.from_numpy(x[start:start + 2])).softmax(1)[:, 1]
            if not torch.isfinite(result).all():
                raise ValueError('Nonfinite reload output')
            scores.extend(result.tolist())
    with (args.report / 'best_validation_predictions.csv').open(newline='') as f:
        reference = {r['sample_id']: float(r['fall_score']) for r in csv.DictReader(f)}
    old = np.array([reference[i] for i in ids]); new = np.array(scores)
    error = float(np.abs(old - new).max()); changed = int(((old >= .5) != (new >= .5)).sum())
    if error > 1e-4 or changed:
        raise ValueError(f'Reloaded model differs: max_error={error}, changed_classes={changed}')
    output = {'new_python_process': True, 'backend': 'cpu', 'records': len(y), 'threshold': .5,
              'max_score_difference_vs_saved_mps_best': error, 'prediction_changes': changed,
              'validation_metrics': metrics(y, scores), 'checkpoint_sha256': sha(args.checkpoint),
              'test_evaluated': False, 'continuous_evaluated': False}
    (args.report / 'independent_reload_check.json').write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
