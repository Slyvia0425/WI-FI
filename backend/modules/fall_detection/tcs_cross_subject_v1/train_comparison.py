"""Train four predeclared GCV/CNN comparison arms; never read test features.

All arms use the saved subject split, balanced sampling, fixed validation loss,
seed42 and threshold0.5. Each arm has its own best/last checkpoint and history.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
MODULE=HERE.parent
sys.path.insert(0,str(MODULE/'tcs_withinsubject_v1'))
from model import CNN
from prepare import sha, RECIPE
sys.path.insert(0,str(MODULE/'continuous_replay_v1'))
from replay import protect


def save_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def save_torch(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp);tmp.replace(path)


def cpu_tree(value):
    if torch.is_tensor(value):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value


def metrics(labels,scores):
    labels=np.asarray(labels);positive=np.asarray(scores)>=.5
    tp=int(((labels==1)&positive).sum());fp=int(((labels==0)&positive).sum())
    fn=int(((labels==1)&~positive).sum());tn=int(((labels==0)&~positive).sum())
    return {'n':len(labels),'fall':int((labels==1).sum()),'nonfall':int((labels==0).sum()),
            'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':tp/(tp+fp) if tp+fp else 0.,
            'recall':tp/(tp+fn) if tp+fn else 0.,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            'accuracy':(tp+tn)/len(labels)}


def state_digest(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def batch(cache,indices,mean,scale,device):
    x=np.array(cache[indices],dtype=np.float32,copy=True)
    if mean is not None:x=(x-mean)/scale
    return torch.from_numpy(x).to(device)


@torch.inference_mode()
def validate(model,cache,rows,indices,mean,scale,device):
    model.eval();scores=[];losses=[]
    for start in range(0,len(indices),32):
        ix=indices[start:start+32]
        y=torch.tensor([rows[i]['label'] for i in ix],dtype=torch.long,device=device)
        logits=model(batch(cache,ix,mean,scale,device))
        if not torch.isfinite(logits).all():raise ValueError('Nonfinite validation logits')
        losses.extend(F.cross_entropy(logits,y,reduction='none').cpu().tolist())
        scores.extend(logits.softmax(1)[:,1].cpu().tolist())
    losses=np.asarray(losses);scores=np.asarray(scores)
    labels=np.array([rows[i]['label'] for i in indices])
    sa=np.array([rows[i]['kind']=='single_action' for i in indices]);fall=sa&(labels==1);nonfall=sa&(labels==0);ct=~sa
    if not all(mask.any() for mask in (fall,nonfall,ct)):raise ValueError('Empty validation group')
    weighted=.5*losses[fall].mean()+.25*losses[nonfall].mean()+.25*losses[ct].mean()
    return {'selection_loss':float(weighted),'fall_sa_loss':float(losses[fall].mean()),
            'nonfall_sa_loss':float(losses[nonfall].mean()),'ct_background_loss':float(losses[ct].mean()),
            'single_action':metrics(labels[sa],scores[sa]),'all_windows':metrics(labels,scores),
            'background_positive_windows':int((scores[ct]>=.5).sum()),'background_windows':int(ct.sum()),
            'background_positive_window_rate':float((scores[ct]>=.5).mean())},scores


def sample_epoch(rows,protocol,epoch,include_ct):
    pos=np.array([i for i,r in enumerate(rows) if r['role']=='train' and r['kind']=='single_action' and r['label']==1])
    neg=np.array([i for i,r in enumerate(rows) if r['role']=='train' and r['kind']=='single_action' and r['label']==0])
    groups={}
    for i,r in enumerate(rows):
        if r['role']=='train' and r['kind']=='continuous_background':groups.setdefault(r['source_member'],[]).append(i)
    groups=[groups[k] for k in sorted(groups)]
    schedule=[]
    for step in range(protocol['steps_per_epoch']):
        generators=[np.random.default_rng(np.random.SeedSequence([protocol['seed'],epoch,step,j])) for j in range(4)]
        positives=generators[0].choice(pos,4,replace=True).tolist()
        negatives=generators[1].choice(neg,4,replace=True).tolist()
        if include_ct:
            continuous=[]
            for _ in range(2):
                group=groups[int(generators[2].integers(len(groups)))]
                continuous.append(group[int(generators[2].integers(len(group)))])
            negatives=negatives[:2]+continuous
        indices=np.array(positives+negatives,dtype=np.int64)
        indices=indices[generators[3].permutation(8)]
        assert sum(rows[i]['label'] for i in indices)==4
        schedule.append(indices)
    return np.stack(schedule)


def train_arm(arm,args,protocol,cache,rows,base_config):
    out=args.output/arm['name'];report=args.report/arm['name']
    if (report/'summary.json').exists():
        previous=json.loads((report/'summary.json').read_text())
        if previous['protocol_sha256']!=base_config['protocol_sha256'] or sha(out/'best.pt')!=previous['checkpoint_sha256']:
            raise ValueError('Completed arm differs from pinned configuration')
        print(json.dumps({'arm':arm['name'],'already_complete':True}),flush=True);return previous
    if out.exists() and not args.resume:raise FileExistsError('Existing arm; use --resume with unchanged source/config')
    out.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
    norm=None;mean=scale=None
    if arm['train_fitted_standardization']:
        name='normalization_sa_ct.json' if arm['continuous_negatives'] else 'normalization_sa.json'
        norm=json.loads((args.features/name).read_text())
        if norm['feature_cache_sha256']!=base_config['feature_cache_sha256'] or norm['fit_role']!='train':raise ValueError('Normalization provenance mismatch')
        mean=np.asarray(norm['mean'],dtype=np.float32);scale=np.asarray(norm['scale'],dtype=np.float32)
    random.seed(protocol['seed']);np.random.seed(protocol['seed']);torch.manual_seed(protocol['seed']);torch.mps.manual_seed(protocol['seed'])
    model=CNN();initial_hash=state_digest(model);model=model.to('mps')
    optimizer=torch.optim.Adam(model.parameters(),lr=protocol['learning_rate'])
    config={**base_config,'arm':arm,'initial_weights_sha256':initial_hash,'normalization_parameters':norm,
            'backend':'mps','dtype':'float32','validation_batch_size':32,'protocol':protocol}
    history=[];best_loss=float('inf');best_epoch=0;stale=0;start_epoch=1
    if args.resume and (out/'last.pt').exists():
        old=json.loads((out/'config.json').read_text())
        if old!=config:raise ValueError('Resume source/config mismatch')
        last=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(last['model_state_dict']);optimizer.load_state_dict(last['optimizer_state_dict'])
        torch.set_rng_state(last['cpu_rng']);torch.mps.set_rng_state(last['mps_rng'])
        history=last['history'];best_loss=last['best_loss'];best_epoch=last['best_epoch'];stale=last['stale'];start_epoch=last['epoch']+1
    else:
        save_json(out/'config.json',config);save_json(report/'config.json',config)
    val_indices=[i for i,r in enumerate(rows) if r['role']=='validation']
    if {rows[i]['subject'] for i in val_indices}!={3}:raise ValueError('Unexpected validation subjects')
    for epoch in range(start_epoch,protocol['max_epochs']+1):
        if stale>=protocol['early_stop_patience']:break
        if shutil.disk_usage(out).free<1024**3:raise RuntimeError('Less than1GiB free; preserve checkpoint and free space before resume')
        begin=time.perf_counter();model.train();total_loss=0.;correct=0
        schedule=sample_epoch(rows,protocol,epoch,arm['continuous_negatives'])
        for step,ix in enumerate(schedule):
            y=torch.tensor([rows[i]['label'] for i in ix],dtype=torch.long,device='mps')
            optimizer.zero_grad(set_to_none=True)
            logits=model(batch(cache,ix,mean,scale,'mps'));loss=F.cross_entropy(logits,y)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward()
            if epoch==start_epoch and step==0:
                if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise ValueError('Nonfinite gradients')
            optimizer.step();total_loss+=float(loss.item())*8;correct+=int((logits.argmax(1)==y).sum().item())
        val,scores=validate(model,cache,rows,val_indices,mean,scale,'mps')
        improved=val['selection_loss']<best_loss
        if improved:best_loss=val['selection_loss'];best_epoch=epoch;stale=0
        else:stale+=1
        record={'epoch':epoch,'train_loss':total_loss/(len(schedule)*8),'train_accuracy':correct/(len(schedule)*8),
                'val_loss':val['selection_loss'],'val_sa_precision':val['single_action']['precision'],
                'val_sa_recall':val['single_action']['recall'],'val_sa_f1':val['single_action']['f1'],
                'val_background_positive_window_rate':val['background_positive_window_rate'],
                'draws_sha256':hashlib.sha256(schedule.tobytes()).hexdigest(),'epoch_seconds':time.perf_counter()-begin}
        history.append(record)
        weights=cpu_tree(model.state_dict())
        if improved:
            checkpoint={'model_state_dict':weights,'model_config':{'input_shape':[1,400,90],'classes':2},
                        'preprocess_config':RECIPE,'normalization_parameters':norm,'label_mapping':{'Nonfall':0,'Fall':1},
                        'seed':protocol['seed'],'best_epoch':epoch,'threshold':.5,'model_version':'tcs_cross_subject_'+arm['name']+'_seed42_v1',
                        'arm':arm,'data_version':{'feature_cache_sha256':base_config['feature_cache_sha256'],
                                                'sample_manifest_sha256':base_config['sample_manifest_sha256']},
                        'split_identifiers':{'train_subjects':[1,2],'validation_subjects':[3],'test_subjects':[4]},
                        'best_validation':val,'run_config':config}
            save_torch(out/'best.pt',checkpoint)
            temp=report/'best_validation_predictions.csv.tmp'
            with temp.open('w',newline='') as f:
                writer=csv.writer(f);writer.writerow(['cache_index','sample_id','kind','label','fall_score','prediction'])
                writer.writerows((i,rows[i]['sample_id'],rows[i]['kind'],rows[i]['label'],float(s),int(s>=.5)) for i,s in zip(val_indices,scores))
            temp.replace(report/'best_validation_predictions.csv')
        save_torch(out/'last.pt',{'model_state_dict':weights,'optimizer_state_dict':cpu_tree(optimizer.state_dict()),
                                 'cpu_rng':torch.get_rng_state(),'mps_rng':torch.mps.get_rng_state(),
                                 'history':history,'epoch':epoch,'best_loss':best_loss,'best_epoch':best_epoch,'stale':stale})
        with (report/'history.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(record));writer.writeheader();writer.writerows(history)
        progress={'arm':arm['name'],'epoch':epoch,'best_epoch':best_epoch,'stale_epochs':stale,**{k:v for k,v in record.items() if k!='draws_sha256'}}
        save_json(report/'progress.json',progress);save_json(args.report/'progress.json',progress)
        print(json.dumps(progress),flush=True)
    best=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
    result={'arm':arm['name'],'epochs_run':len(history),'best_epoch':best['best_epoch'],
            'stop_reason':'patience10' if stale>=protocol['early_stop_patience'] else 'max_epochs100',
            'best_validation':best['best_validation'],'threshold':.5,'checkpoint_sha256':sha(out/'best.pt'),
            'protocol_sha256':base_config['protocol_sha256'],'initial_weights_sha256':initial_hash,
            'elapsed_epoch_seconds':sum(r['epoch_seconds'] for r in history),'test_evaluated':False,
            'preserved_artifacts':protect()}
    save_json(report/'summary.json',result)
    print(json.dumps({'completed_arm':arm['name'],**result}),flush=True)
    del model,optimizer,best,weights;torch.mps.empty_cache()
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--features',type=Path,required=True);ap.add_argument('--protocol',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--report',type=Path,required=True)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no hidden backend change')
    if shutil.disk_usage(args.features).free<5*1024**3:raise RuntimeError('Reserve5GiB before formal training')
    protocol=json.loads(args.protocol.read_text());prep=json.loads((args.features/'preparation.json').read_text())
    rows=json.loads((args.features/'samples.json').read_text())
    if sha(args.features/'features.f32')!=protocol['feature_cache_sha256'] or prep['feature_cache_sha256']!=protocol['feature_cache_sha256']:
        raise ValueError('Feature cache differs from predeclared protocol')
    if sha(args.features/'samples.json')!=prep['sample_manifest_sha256']:raise ValueError('Sample manifest changed')
    if {r['subject'] for r in rows}!={1,2,3}:raise ValueError('Unexpected subjects in prepared features')
    args.output.mkdir(parents=True,exist_ok=True);args.report.mkdir(parents=True,exist_ok=True)
    base={'protocol_sha256':sha(args.protocol),'feature_cache_sha256':prep['feature_cache_sha256'],
          'sample_manifest_sha256':prep['sample_manifest_sha256'],'training_source_sha256':sha(Path(__file__)),
          'model_source_sha256':sha(MODULE/'tcs_withinsubject_v1/model.py'),
          'versions':{'torch':str(torch.__version__),'numpy':str(np.__version__)},'protected_before':protect()}
    save_json(args.report/'execution_config.json',base)
    cache=np.memmap(args.features/'features.f32',dtype=np.float32,mode='r',shape=tuple(prep['shape']))
    results=[]
    for arm in protocol['arms']:results.append(train_arm(arm,args,protocol,cache,rows,base))
    if len({r['initial_weights_sha256'] for r in results})!=1:raise ValueError('Initial states differ between arms')
    save_json(args.report/'training_comparison.json',{'arms':results,'same_initial_weights_verified':True,'test_evaluated':False})
    print('All four training arms completed.',flush=True)


if __name__=='__main__':main()
