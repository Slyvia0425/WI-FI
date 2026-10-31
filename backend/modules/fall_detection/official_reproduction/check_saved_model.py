"""Independent CPU reload of official MLP/ResNet best checkpoint; validation only."""
import argparse,csv,json,hashlib,sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from run_baseline import BenchmarkCSIDataset,MLPClassifier,ResNet18Classifier,strict_collate,device_metrics,HERE,MODULE


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run-name',default='official_mlp_seed42_sourcefaithful_v1');args=ap.parse_args()
 report=MODULE/'reports'/args.run_name;ck=MODULE/'outputs'/args.run_name
 c=torch.load(ck/'reproduction_bundle.pt',map_location='cpu',weights_only=True)
 torch.set_num_threads(2)
 model=MLPClassifier() if c['model']=='MLPClassifier' else ResNet18Classifier()
 model.load_state_dict(c['model_state_dict'],strict=True);model.eval()
 val=BenchmarkCSIDataset('/Users/chenxiaohe/Documents/CSI-Bench-v12','FallDetection','val_id')
 assert val.split_metadata['id'].tolist()==[r['sample_id'] for r in csv.DictReader((report/'best_validation_predictions.csv').open())]
 logits=[];loss=0.
 with torch.no_grad():
  for x,y in DataLoader(val,batch_size=128,shuffle=False,collate_fn=strict_collate):
   z=model(x);logits.append(z);loss+=float(torch.nn.functional.cross_entropy(z,y))*len(y)
 z=torch.cat(logits);assert len(z)==1005 and torch.isfinite(z).all()
 y=(val.split_metadata['label']=='Fall').astype(int).to_numpy();pred=(z.argmax(1).numpy()==0).astype(int)
 scores=z.softmax(1)[:,0].tolist();saved=list(csv.DictReader((report/'best_validation_predictions.csv').open()))
 diff=max(abs(s-float(r['fall_score'])) for s,r in zip(scores,saved))
 changed=sum((s>=.5)!=(float(r['fall_score'])>=.5) for s,r in zip(scores,saved))
 assert diff<1e-4 and changed==0 and abs(loss/1005-c['best_val_loss'])<1e-5
 by={}
 for d in ('HP','ESP32'):
  mask=(val.split_metadata['device']==d).to_numpy();by[d]=device_metrics(y[mask],pred[mask])
 r={'new_python_process':True,'device':'cpu','best_epoch':c['best_epoch'],'validation_loss':loss/1005,
    'max_score_difference':diff,'classifications_changed':changed,'all':device_metrics(y,pred),'by_device':by,'test_evaluated':False,
    'upstream_source_hashes_match':all(hashlib.sha256((HERE/'upstream'/n).read_bytes()).hexdigest()==v for n,v in json.loads((HERE/'upstream_provenance.json').read_text())['files_sha256'].items())}
 assert r['upstream_source_hashes_match']
 (report/'checkpoint_check.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))

if __name__=='__main__':main()
