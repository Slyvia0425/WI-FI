"""Evaluate ONLY an integrity-checked frozen pipeline; never selects models or thresholds."""
import argparse,csv,json
from pathlib import Path
from runtime import load_frozen,predict_files,sha
from compare_thresholds import metrics


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--splits',default='test_id,test_easy,test_medium,test_hard')
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args()
    if args.output_dir.exists():raise FileExistsError('Fresh evaluation directory required')
    model,state,kind=load_frozen(args.checkpoint)
    assert sha(args.manifest)==state['manifest_sha256'], 'Manifest changed after model selection'
    manifest=list(csv.DictReader(args.manifest.open()))
    # Each physical segment is inferred once, even when easy/medium/hard are subsets of test_id.
    names=args.splits.split(',')
    if any(s not in ('val_id','test_id','test_easy','test_medium','test_hard') for s in names):raise ValueError('Unexpected split')
    needed=[r for r in manifest if set(r['split'].split(';'))&set(names)]
    assert len({r['sample_id'] for r in needed})==len(needed)
    if any(set(r['split'].split(';'))&{'train_id','val_id'} for r in needed) and 'val_id' not in names:raise ValueError('Unexpected training/validation membership')
    predictions=predict_files(model,state,kind,needed,batch_size=32 if kind=='original_1d_cnn' else 8)
    by_id={r['sample_id']:r for r in predictions};results={}
    for split in names:
        rows=[r for r in manifest if split in r['split'].split(';')]
        records=[by_id[r['sample_id']] for r in rows]
        results[split]={'all':metrics(records,state['selected_threshold']),'by_device':{d:metrics([r for r in records if r['device']==d],state['selected_threshold']) for d in ('HP','ESP32') if any(r['device']==d for r in records)}}
    if set(names)=={'test_id','test_easy','test_medium','test_hard'}:
        sets={s:{r['sample_id'] for r in manifest if s in r['split'].split(';')} for s in names}
        assert not sets['test_easy']&sets['test_medium'] and not sets['test_easy']&sets['test_hard'] and not sets['test_medium']&sets['test_hard']
        assert sets['test_id']==sets['test_easy']|sets['test_medium']|sets['test_hard']
    args.output_dir.mkdir(parents=True)
    with (args.output_dir/'predictions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(predictions[0]));w.writeheader();w.writerows(predictions)
    result={'candidate_kind':kind,'seed':state['seed'],'threshold':state['selected_threshold'],
            'checkpoint_sha256':sha(args.checkpoint),'frozen_receipt_sha256':sha(args.checkpoint.with_suffix('.json')),
            'unique_segments_inferred':len(predictions),'sets':results,'threshold_or_model_changed':False,
            'subset_note':'easy,medium,hard partition test_id; they are not four independent test datasets',
            'known_limitations':state['known_limitations'],'deployment_claim':state['deployment_claim']}
    (args.output_dir/'evaluation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
