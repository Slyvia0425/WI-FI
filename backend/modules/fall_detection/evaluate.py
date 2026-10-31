"""Evaluate named official splits with a frozen checkpoint and threshold; no tuning."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from torch.utils.data import DataLoader

if __package__:
    from .dataset import FallDataset
    from .inference_runtime import collect_predictions, load_model, record_metrics, sha256
else:
    from dataset import FallDataset
    from inference_runtime import collect_predictions, load_model, record_metrics, sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--splits',default='test_id,test_easy,test_medium,test_hard')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    allowed={'train_id','val_id','test_id','test_easy','test_medium','test_hard'}
    splits=args.splits.split(',')
    if not splits or len(set(splits))!=len(splits) or not set(splits)<=allowed:
        raise ValueError('Invalid/duplicate official split names')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if (args.output_dir/'evaluation.json').exists():
        raise FileExistsError('Evaluation already exists; use a new output directory')
    initial_hash=sha256(args.checkpoint)
    model,state=load_model(args.checkpoint,'cpu')
    threshold=state['selected_threshold']
    result={'checkpoint_sha256':initial_hash,'model_version':state['model_version'],
            'best_epoch':state['best_epoch'],'threshold':threshold,'device':'cpu',
            'started_at_utc':datetime.now(timezone.utc).isoformat(),'splits':{},
            'warning':'Official content duplicates and device/difficulty confounding remain; difficulty subsets overlap test_id.'}
    # Cache identical ID predictions within this evaluation, so nested difficulty sets use identical scores.
    cache={}
    for split in splits:
        dataset=FallDataset(args.manifest,state['preprocess_config'],split)
        if split in state['split_identifiers'] and {r['sample_id'] for r in dataset.rows}!=set(state['split_identifiers'][split]):
            raise ValueError('Split differs from frozen experiment')
        if all(row['sample_id'] in cache for row in dataset.rows):
            records=[cache[row['sample_id']] for row in dataset.rows]
        else:
            records=collect_predictions(model,DataLoader(dataset,batch_size=32,shuffle=False),'cpu')
            cache.update({r['sample_id']:r for r in records})
        for row in records:
            row['prediction']='Fall' if row['fall_score']>=threshold else 'Nonfall'
        with (args.output_dir/(split+'_predictions.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        devices=sorted({r['device'] for r in records})
        result['splits'][split]={'overall':record_metrics(records,threshold),
                                'by_device':{d:record_metrics([r for r in records if r['device']==d],threshold) for d in devices}}
        print(split,json.dumps(result['splits'][split]),flush=True)
    if sha256(args.checkpoint)!=initial_hash:
        raise ValueError('Checkpoint changed during evaluation')
    result['completed_at_utc']=datetime.now(timezone.utc).isoformat()
    result['checkpoint_unchanged']=True
    (args.output_dir/'evaluation.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
