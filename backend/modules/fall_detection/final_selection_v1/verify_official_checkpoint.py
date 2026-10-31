"""Independent-process validation replay for recovered official ResNet18 weights."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment-root',type=Path,required=True)
    ap.add_argument('--run-name',required=True)
    ap.add_argument('--upstream',type=Path,required=True)
    ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=ap.parse_args();sys.path.insert(0,str(args.upstream));torch.set_num_threads(2)
    from model.supervised.models import ResNet18Classifier
    from load.supervised.benchmark_dataset import BenchmarkCSIDataset
    report=args.experiment_root/'reports'/args.run_name
    ckpath=args.experiment_root/'outputs'/args.run_name/'reproduction_bundle.pt'
    ck=torch.load(ckpath,map_location='cpu',weights_only=True)
    assert ck['model']=='ResNet18Classifier' and ck['label_mapping']=={'Fall':0,'Nonfall':1}
    model=ResNet18Classifier(win_len=500,feature_size=232,num_classes=2,in_channels=1)
    model.load_state_dict(ck['model_state_dict'],strict=True);model.to(args.device).eval()
    val=BenchmarkCSIDataset(str(args.data_root),'FallDetection',split_name='val_id',file_format='h5',data_key='CSI_amps')
    assert sorted(val.split_metadata['id'].tolist())==ck['split_identifiers']['val_id']
    records=list(csv.DictReader((report/'best_validation_predictions.csv').open()))
    assert [r['sample_id'] for r in records]==val.split_metadata['id'].astype(str).tolist()
    scores=[]
    with torch.inference_mode():
        for x,y in DataLoader(val,batch_size=128,shuffle=False,num_workers=0):
            assert torch.isfinite(x).all();out=model(x.to(args.device))
            assert torch.isfinite(out).all();scores.extend(out.softmax(1)[:,0].cpu().tolist())
    original=np.array([float(r['fall_score']) for r in records]);reloaded=np.array(scores)
    difference=float(np.max(np.abs(original-reloaded)))
    changed=int(((original>=.5)!=(reloaded>=.5)).sum())
    assert difference<1e-4 and changed==0,(difference,changed)
    result={'new_python_process':True,'validation_n':len(scores),'max_score_difference':difference,
            'changed_predictions_at_0_5':changed,'best_epoch':ck['best_epoch'],'test_evaluated':False,
            'bundle_sha256':hashlib.sha256(ckpath.read_bytes()).hexdigest(),'bundle_bytes':ckpath.stat().st_size}
    (report/'independent_reload_check.json').write_text(json.dumps(result,indent=2))
    print('INDEPENDENT_RELOAD_PASS',json.dumps(result),flush=True)

if __name__=='__main__':main()
