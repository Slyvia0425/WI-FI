"""New-process candidate loading and exact official validation replay only."""
import argparse,csv,json,importlib.metadata
from pathlib import Path
from runtime import load_candidate,predict_files,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--kind',choices=['original_1d_cnn','official_resnet18'],required=True)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args()
    if args.output_dir.exists():raise FileExistsError('Fresh output directory required')
    model,state=load_candidate(args.checkpoint,args.kind)
    if args.kind=='original_1d_cnn':assert sha(args.manifest)==state['manifest_sha256']
    rows=[r for r in csv.DictReader(args.manifest.open()) if 'val_id' in r['split'].split(';')]
    assert len(rows)==1005
    assert {r['sample_id'] for r in rows}==set(state['split_identifiers']['val_id'])
    records=predict_files(model,state,args.kind,rows,batch_size=32 if args.kind=='original_1d_cnn' else 8)
    args.output_dir.mkdir(parents=True)
    out=args.output_dir/'validation_predictions.csv'
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    receipt={'kind':args.kind,'seed':state['seed'],'best_epoch':state['best_epoch'],'n':len(records),
             'checkpoint_sha256':sha(args.checkpoint),'predictions_sha256':sha(out),'device':'cpu',
             'test_evaluated':False,'new_python_process':True,
             'inference_versions':{n:importlib.metadata.version(n) for n in ('torch','numpy','h5py')}}
    (args.output_dir/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
