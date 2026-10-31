"""Select maximum Fall F1 threshold using validation only; freeze before test evaluation."""
import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__:
    from .dataset import FallDataset
    from .inference_runtime import collect_predictions, load_model, record_metrics, sha256
else:
    from dataset import FallDataset
    from inference_runtime import collect_predictions, load_model, record_metrics, sha256


def select_threshold(records):
    # Every distinct score group gives one attainable nonempty positive set.
    # Use interval midpoints, rather than placing the threshold on a sample score.
    # With positive validation examples an all-negative point (F1=0) cannot be optimal.
    scores = np.array([r['fall_score'] for r in records], dtype=np.float64)
    labels = np.array([r['label'] for r in records], dtype=np.int64)
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('Invalid validation scores')
    if not 0 < labels.sum() < len(labels):
        raise ValueError('Validation must contain both verified classes')
    unique = np.unique(scores)
    thresholds = [0.] + ((unique[:-1]+unique[1:])/2).tolist()
    rows = []
    best_key, best = None, None
    for threshold in thresholds:
        result = record_metrics(records, threshold)
        key = (Fraction(2*result['tp'], 2*result['tp']+result['fp']+result['fn']),
               Fraction(result['tp'], result['fall']), threshold)
        row = dict(threshold=threshold, **result)
        rows.append(row)
        if best_key is None or key > best_key:
            best_key, best = key, row
    return best, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-checkpoint', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_checkpoint.exists() or args.output_checkpoint.with_suffix('.json').exists():
        raise FileExistsError('Frozen artifacts already exist; refusing to overwrite')
    model, state = load_model(args.checkpoint, device='cpu', require_frozen=False)
    val = FallDataset(args.manifest, state['preprocess_config'], 'val_id')
    if {r['sample_id'] for r in val.rows} != set(state['split_identifiers']['val_id']):
        raise ValueError('Validation membership differs from training checkpoint')
    predictions = collect_predictions(model, DataLoader(val, batch_size=32, shuffle=False), 'cpu')
    best, candidates = select_threshold(predictions)
    # Same process, fresh architecture + state reload: eval scores must match exactly here.
    reloaded, _ = load_model(args.checkpoint, device='cpu', require_frozen=False)
    repeated = collect_predictions(reloaded, DataLoader(val, batch_size=32, shuffle=False), 'cpu')
    difference = max(abs(a['fall_score']-b['fall_score']) for a,b in zip(predictions,repeated))
    if difference > 1e-7:
        raise ValueError('Reload changed validation scores')
    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    prediction_path = args.report_dir/'frozen_validation_predictions.csv'
    for path, rows in [(prediction_path,predictions),(args.report_dir/'threshold_candidates.csv',candidates)]:
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    state.update(frozen=True, selected_threshold=best['threshold'],
                 threshold_selection_status='validation_f1_frozen',
                 threshold_selection={'split':'val_id','objective':'max Fall F1',
                                      'tie_break':'higher Recall, then higher threshold',
                                      'validation_result':best,'prediction_rule':'fall_score >= threshold',
                                      'candidate_rule':'midpoints between unique scores plus 0; all-negative F1=0 cannot win',
                                      'validation_predictions_sha256':sha256(prediction_path)},
                 frozen_at_utc=datetime.now(timezone.utc).isoformat(), inference_device='cpu')
    torch.save(state,args.output_checkpoint)
    source_files=['freeze_model.py','inference_runtime.py','evaluate.py','infer.py','dataset.py','preprocess.py','models/cnn.py']
    receipt={'model_version':state['model_version'],'best_epoch':state['best_epoch'],
             'selected_threshold':best['threshold'],'threshold_selection':state['threshold_selection'],
             'frozen_at_utc':state['frozen_at_utc'],'checkpoint_sha256':sha256(args.output_checkpoint),
             'source_training_checkpoint_sha256':sha256(args.checkpoint),'manifest_sha256':state['manifest_sha256'],
             'data_version':state['data_version'],'test_used_for_selection':False,'inference_device':'cpu',
             'source_sha256':{f:sha256(Path(__file__).parent/f) for f in source_files}}
    args.output_checkpoint.with_suffix('.json').write_text(json.dumps(receipt,indent=2))
    result={'selected':best,'validation_at_0_5':record_metrics(predictions,.5),
            'candidate_count':len(candidates),'same_process_reload_max_score_difference':difference,
            'frozen_checkpoint':str(args.output_checkpoint),'receipt':receipt}
    (args.report_dir/'threshold_selection.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
