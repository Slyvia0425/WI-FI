"""Pinned official baseline/Dataset/TaskTrainer, instrumented train/val only.

No test Dataset is constructed. Scheduler advancement preserves upstream code,
including its batch-count/epoch-count mismatch. This is not the corrected variant.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import random
import sys
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, f1_score

HERE=Path('/content/csi_bench_official')
import os
MODULE=Path(os.environ.get('FALL_EXPERIMENT_ROOT','/content/drive/MyDrive/CSI-Bench_FallDetection_FinalSelection_v1'))
sys.path.insert(0,str(HERE))
from model.supervised.models import ResNet18Classifier, MLPClassifier
from load.supervised.benchmark_dataset import BenchmarkCSIDataset
from engine.supervised.task_trainer import TaskTrainer


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def save(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)


def strict_collate(batch):
    if any(v is None for v in batch):raise ValueError('Official loader returned a bad record; refusing silent skip')
    return torch.utils.data.default_collate(batch)


def device_metrics(y,p):
    precision,recall,f1,_=precision_recall_fscore_support(y,p,average='binary',zero_division=0)
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel().tolist()
    return dict(n=len(y),fall=int(sum(y)),nonfall=int(len(y)-sum(y)),precision=float(precision),
                recall=float(recall),fall_f1=float(f1),weighted_f1=float(f1_score(y,p,average='weighted',zero_division=0)),
                tp=tp,fp=fp,fn=fn,tn=tn,false_positive_rate=fp/(fp+tn) if fp+tn else 0.)


class ObservedTrainer(TaskTrainer):
    """Only observation and strict coverage checks; super() performs upstream math."""
    def evaluate(self,loader):
        captured=[]
        handle=self.model.register_forward_hook(lambda m,i,o:captured.append(o.detach().cpu()))
        try:loss,acc=super().evaluate(loader)
        finally:handle.remove()
        logits=torch.cat(captured)
        if len(logits)!=len(loader.dataset) or not torch.isfinite(logits).all():
            raise ValueError('Incomplete/nonfinite official validation outputs')
        meta=loader.dataset.split_metadata
        y=(meta['label']=='Fall').astype(int).to_numpy()
        fallidx=loader.dataset.label_mapper.label_to_idx['Fall']
        pred=(logits.argmax(1).numpy()==fallidx).astype(int)
        score=logits.softmax(1)[:,fallidx].tolist()
        records=[dict(sample_id=str(row['id']),device=str(row['device']),label=int(label),fall_score=float(s))
                 for (_,row),label,s in zip(meta.iterrows(),y,score)]
        self.observed_records=records
        epoch=self.current_epoch+1
        summary={'epoch':epoch,'val_loss':loss,'accuracy':acc,
                 'learning_rate_used':self.optimizer.param_groups[0]['lr'],
                 'all':device_metrics(y,pred),'by_device':{}}
        for d in ('HP','ESP32'):
            mask=(meta['device']==d).to_numpy();summary['by_device'][d]=device_metrics(y[mask],pred[mask])
        improved=loss<getattr(self,'observed_best',float('inf'))
        if improved:
            self.observed_best=loss;self.observed_best_epoch=epoch
            with (self.report_path/'best_validation_predictions.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
            save(self.report_path/'best_validation_metrics.json',summary)
            # Extra atomic weight backup: protects against interruption while upstream
            # writes its optimizer-bearing best_model.pt. Does not affect optimization.
            durable=self.report_path.parent.parent/'outputs'/self.report_path.name/'durable_best_weights.pt'
            tmp=durable.with_suffix('.pt.tmp')
            torch.save({'model_state_dict':{k:v.detach().cpu() for k,v in self.model.state_dict().items()},
                        'epoch':self.current_epoch,'val_loss':loss},tmp)
            tmp.replace(durable)
            assert durable.is_file() and durable.stat().st_size>0
            print('DURABLE_BEST',str(durable),'epoch',epoch,flush=True)
        self.observed_history.append(summary)
        save(self.report_path/'validation_history.json',self.observed_history)
        save(self.report_path/'progress.json',{'completed':False,'last_epoch':epoch,
            'best_epoch':self.observed_best_epoch,'best_val_loss':self.observed_best,
            'wait':epoch-self.observed_best_epoch,'current':summary})
        print('OBSERVED',json.dumps(summary),flush=True)
        return loss,acc


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',type=Path,default=Path('/content/CSI-Bench-v12'))
    ap.add_argument('--model',choices=['mlp','resnet18'],default='resnet18')
    ap.add_argument('--run-name',default='official_resnet18_seed42_recovery_v1')
    ap.add_argument('--preflight-only',action='store_true')
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    report=MODULE/'reports'/args.run_name;ckdir=MODULE/'outputs'/args.run_name
    if (report/'progress.json').exists() or (ckdir/'best_model.pt').exists():raise FileExistsError('Choose a fresh run name')
    report.mkdir(parents=True,exist_ok=True);ckdir.mkdir(parents=True,exist_ok=True)
    import subprocess
    commit='b1beac7b0fa46e01748901195ecfbe309dfc0180'
    assert subprocess.check_output(['git','-C',str(HERE),'rev-parse','HEAD'],text=True).strip()==commit
    assert not subprocess.check_output(['git','-C',str(HERE),'diff','--name-only'],text=True).strip()
    source={'commit':commit}
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(2)
    assert torch.cuda.is_available(), 'This run requires a CUDA GPU'
    device=torch.device('cuda:0');torch.cuda.manual_seed_all(args.seed)
    # Upstream seed helper does not force deterministic algorithms; preserve that.
    datasets={s:BenchmarkCSIDataset(str(args.data_root),'FallDetection',split_name=s,file_format='h5',data_key='CSI_amps') for s in ('train_id','val_id')}
    train,val=datasets.values();assert len(train)==4690 and len(val)==1005
    assert train.label_mapper.label_to_idx==val.label_mapper.label_to_idx=={'Fall':0,'Nonfall':1}
    assert not set(train.split_metadata['id'])&set(val.split_metadata['id'])
    # Do not construct or evaluate any test Dataset. CRCs were checked on download.
    for d in datasets.values():
        assert set(d.split_metadata['id'])==d.split_ids
        assert not d.missing_file_paths
    for d in datasets.values():
        for i in range(len(d)):
            item=d[i]
            if item is None or item[0].shape!=(1,500,232) or not torch.isfinite(item[0]).all():raise ValueError('Data preflight failed')
    # Explicitly demonstrate official normalization/padding on one real record/device.
    import h5py
    samples={}
    for d in ('HP','ESP32'):
        i=int(train.split_metadata.index[train.split_metadata['device']==d][0]);x,y=train[i]
        row=train.split_metadata.iloc[i]
        with h5py.File(args.data_root/'FallDetection'/row['file_path'],'r') as f:raw=torch.from_numpy(f['CSI_amps'][()]).float().permute(2,1,0)
        expected=(raw-raw.mean(dim=(1,2),keepdim=True))/raw.std(dim=(1,2),keepdim=True).clamp_min(1e-8)
        torch.testing.assert_close(x[:,:,:raw.shape[-1]],expected,rtol=0,atol=0)
        assert torch.count_nonzero(x[:,:,raw.shape[-1]:])==0
        samples[d]={'sample_id':str(row['id']),'shape':list(x.shape),'official_label':int(y),
                    'valid_mean':float(expected.mean()),'valid_std':float(expected.std()),'padded_nonzero':0}
    model=(MLPClassifier(win_len=500,feature_size=232,num_classes=2) if args.model=='mlp' else ResNet18Classifier(win_len=500,feature_size=232,num_classes=2,in_channels=1)).to(device)
    model.eval()
    with torch.no_grad():out=model(torch.stack([train[0][0],train[1][0]]).to(device))
    assert out.shape==(2,2) and torch.isfinite(out).all()
    trainloader=DataLoader(train,batch_size=128,shuffle=True,num_workers=0,pin_memory=False,collate_fn=strict_collate)
    valloader=DataLoader(val,batch_size=128,shuffle=False,num_workers=0,pin_memory=False,collate_fn=strict_collate)
    config={'source_commit':source['commit'],'model':model.__class__.__name__, 'pretrained_weights':None,'seed':args.seed,
        'batch_size':128,'max_epochs':100,'patience':15,'optimizer':'AdamW','lr':.001,'weight_decay':1e-5,
        'criterion':'unweighted CrossEntropyLoss','label_mapping':train.label_mapper.label_to_idx,
        'train_n':len(train),'val_n':len(val),'test_evaluated':False,'device':str(device),
        'gpu':torch.cuda.get_device_name(0),
        'platform_deviations':['Colab Tesla T4 instead of A10G','num_workers=0 instead of 4 (explicit Colab adaptation)',
            'strict failure on bad data instead of silently filtering None; all train/val files passed'],
        'scheduler_caveat':'upstream batch-count warmup lambda advanced once per epoch; retained unchanged',
        'warmup_steps':len(trainloader)*5,'scheduler_updates_per_epoch':1,'preflight':samples,
        'data_version':12,'data_receipt':json.loads((args.data_root/'download_receipt.json').read_text()),
        'split_sha256':{s:sha(args.data_root/'FallDetection/splits'/f'{s}.json') for s in datasets},
        'known_data_limitation':'one identical train/validation content pair in official v12 split; retained',
        'source_unchanged':True,'persistent_root':str(MODULE),
        'durable_checkpoint':'atomic weights backup each validation improvement plus official best checkpoint',
        'versions':{n:importlib.metadata.version(n) for n in ('torch','torchvision','numpy','h5py','pandas','scikit-learn')}}
    save(report/'run_config.json',config);print('CONFIG',json.dumps(config),flush=True)
    if args.preflight_only:return
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-5)
    num_steps=len(trainloader)*100;warmup_steps=len(trainloader)*5
    def official_schedule(step):
        if step<warmup_steps:return float(step)/float(max(1,warmup_steps))
        progress=float(step-warmup_steps)/float(max(1,num_steps-warmup_steps))
        return .5*(1.+math.cos(math.pi*progress))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=official_schedule)
    trainer=ObservedTrainer(model,trainloader,val_loader=valloader,test_loader=None,
        criterion=torch.nn.CrossEntropyLoss(),optimizer=optimizer,scheduler=scheduler,
        device=device,save_path=str(ckdir),num_classes=2,label_mapper=train.label_mapper,
        config={'epochs':100,'patience':15,'warmup_epochs':5})
    trainer.report_path=report;trainer.observed_history=[]
    start=time.monotonic();trainer.train()
    checkpoint=torch.load(ckdir/'best_model.pt',map_location='cpu',weights_only=True)
    metrics=json.loads((report/'best_validation_metrics.json').read_text())
    assert checkpoint['epoch']+1==metrics['epoch'] and abs(checkpoint['val_loss']-metrics['val_loss'])<1e-8
    config['model_state_dict']={k:v.cpu() for k,v in checkpoint['model_state_dict'].items()}
    config['best_epoch']=metrics['epoch'];config['best_val_loss']=metrics['val_loss']
    config['split_identifiers']={s:sorted(d.split_metadata['id'].tolist()) for s,d in datasets.items()}
    torch.save(config,ckdir/'reproduction_bundle.pt')
    summary={'completed':True,'epochs_run':len(trainer.observed_history),'best_epoch':metrics['epoch'],
        'best_val_loss':metrics['val_loss'],'best_metrics':metrics,'elapsed_seconds':time.monotonic()-start,
        'test_evaluated':False,'prior_local_artifacts_touched':False,'checkpoint_sha256':sha(ckdir/'best_model.pt')}
    save(report/'training_summary.json',summary);save(report/'progress.json',summary)
    history=trainer.observed_history
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    epochs=[h['epoch'] for h in history]
    axes[0].plot(epochs,trainer.train_losses,label='train');axes[0].plot(epochs,[h['val_loss'] for h in history],label='val');axes[0].set_ylabel('Unweighted CE')
    for d,ax in zip(('HP','ESP32'),axes[1:]):
        for k in ('precision','recall','fall_f1','weighted_f1'):ax.plot(epochs,[h['by_device'][d][k] for h in history],label=k)
        ax.set_title(d+' validation');ax.set_ylim(0,1.05)
    for ax in axes:ax.axvline(metrics['epoch'],ls='--',color='gray');ax.legend(fontsize=8);ax.set_xlabel('Epoch');ax.grid(alpha=.2)
    fig.savefig(report/'training_curves.png',dpi=150);plt.close(fig)
    print('DONE',json.dumps(summary),flush=True)

if __name__=='__main__':main()
