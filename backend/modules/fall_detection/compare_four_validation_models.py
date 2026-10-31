"""Replay four best models on official val_id only, threshold 0.5; no optimization."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataset import FallDataset
from models.cnn import FallCNN
from models.sensefi_style import SenseFiStyleCNN,collate_device_segments
from models.cnn_lstm import OriginalCNNLSTM,SenseFiStyleCNNLSTM
from train_cnn_lstm import validate

ROOT=Path(__file__).resolve().parent
SPECS=[('original_cnn','cnn_seed42_mps_v1','stage4_seed42_mps_v1',FallCNN,False),
       ('sensefi_style_cnn','sensefi_style_seed42_mps_v1','sensefi_style_seed42_mps_v1',SenseFiStyleCNN,True),
       ('original_cnn_lstm','original_cnn_lstm_seed42_mps_v1','original_cnn_lstm_seed42_mps_v1',OriginalCNNLSTM,False),
       ('sensefi_style_cnn_lstm','sensefi_style_cnn_lstm_seed42_mps_v1','sensefi_style_cnn_lstm_seed42_mps_v1',SenseFiStyleCNNLSTM,True)]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    out=ROOT/'reports/four_model_validation_seed42_v1'
    if out.exists():raise FileExistsError('Choose a fresh report directory')
    out.mkdir(parents=True)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    result={'threshold':.5,'split':'val_id','compute_device':'cpu','test_evaluated':False,
            'training_performed':False,'models':{}}
    reference=None;joined={};comparison=[]
    for name,ckdir,reportdir,cls,grouped in SPECS:
        path=ROOT/'outputs'/ckdir/'best.pt';hash_before=sha(path)
        c=torch.load(path,map_location='cpu',weights_only=True)
        provenance=(c['manifest_sha256'],c['split_identifiers'],c['preprocess_config'])
        if reference is None:reference=provenance
        else:assert reference==provenance
        assert sha(ROOT/'reports/sample_manifest.csv')==c['manifest_sha256']
        dataset=FallDataset(ROOT/'reports/sample_manifest.csv',c['preprocess_config'],'val_id')
        model=cls(**c['model_config']) if name=='original_cnn' else cls(dropout=c['model_config']['dropout'])
        model.load_state_dict(c['model_state_dict'],strict=True)
        loader=DataLoader(dataset,batch_size=32,shuffle=False,collate_fn=collate_device_segments if grouped else None)
        criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(c['training_config']['pos_weight']))
        loss,metrics,records=validate(model,loader,criterion,torch.device('cpu'))
        assert {r['sample_id'] for r in records}==set(c['split_identifiers']['val_id']) and len(records)==1005
        saved={r['sample_id']:r for r in csv.DictReader((ROOT/'reports'/reportdir/'best_validation_predictions.csv').open())}
        diff=max(abs(r['fall_score']-float(saved[r['sample_id']]['fall_score'])) for r in records)
        changed=sum((r['fall_score']>=.5)!=(float(saved[r['sample_id']]['fall_score'])>=.5) for r in records)
        assert diff<1e-4 and changed==0 and abs(loss-c['best_val_loss'])<1e-5
        assert sha(path)==hash_before
        for r in records:
            sid=r['sample_id']
            if sid not in joined:joined[sid]={'sample_id':sid,'device':r['device'],'label':r['label']}
            assert joined[sid]['device']==r['device'] and joined[sid]['label']==r['label']
            joined[sid][name+'_score']=r['fall_score']
        result['models'][name]={'best_epoch':c['best_epoch'],'validation_loss':loss,'metrics':metrics,
            'max_score_difference_from_mps':diff,'classifications_changed':changed,'checkpoint_sha256':hash_before,
            'trainable_parameters':sum(p.numel() for p in model.parameters())}
        for d in ('all','HP','ESP32'):
            keys=['precision','recall','f1','tp','fp','fn','tn']
            rr={k:metrics[k if d=='all' else d+'_'+k] for k in keys}
            rr.update(model=name,device=d,n=1005 if d=='all' else metrics[d+'_n'])
            rr['false_positive_rate']=rr['fp']/(rr['fp']+rr['tn'])
            comparison.append(rr)
        print('REPLAY',name,'epoch',c['best_epoch'],'loss',loss,'maxdiff',diff,flush=True)
    for _,_,reportdir,_,_ in SPECS[2:]:
        before=json.loads((ROOT/'reports'/reportdir/'preserved_artifacts_before.json').read_text())
        assert all(sha(ROOT/n)==v for n,v in before.items())
    result['previous_models_sources_preprocessing_unchanged']=True
    with (out/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(comparison[0]));w.writeheader();w.writerows(comparison)
    with (out/'validation_predictions.csv').open('w',newline='') as f:
        rows=list(joined.values());w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'comparison.json').write_text(json.dumps(result,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    names=[s[0] for s in SPECS];labels=['Original CNN','New CNN','Original + LSTM','New + LSTM']
    for ax,d in zip(axes,('all','HP','ESP32')):
        rows=[r for r in comparison if r['device']==d]
        x=np.arange(4)
        for i,k in enumerate(('precision','recall','f1')):
            ax.bar(x+(i-1)*.24,[r[k] for r in rows],.24,label=k)
        ax.set_xticks(x,labels,rotation=25,ha='right');ax.set_ylim(0,1.05);ax.set_title(d+' validation');ax.legend(fontsize=8)
    fig.suptitle('Seed 42; each model selected by validation loss; threshold 0.5; no test evaluation')
    fig.savefig(out/'four_model_validation.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,(name,_,rd,_,_) in zip(axes.flat,SPECS):
        rows=list(csv.DictReader((ROOT/'reports'/rd/'history.csv').open()))
        for k in ('train_loss','val_loss'):ax.plot([int(r['epoch']) for r in rows],[float(r[k]) for r in rows],label=k)
        ax.axvline(result['models'][name]['best_epoch'],color='gray',ls='--');ax.set_title(name)
        ax.set_xlabel('Epoch');ax.set_ylabel('Weighted BCE');ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/'four_training_curves.png',dpi=150);plt.close(fig)
    print('DONE',json.dumps(comparison),flush=True)

if __name__=='__main__':main()
