"""Shape/smoke/fixed-subset diagnostics only; loads official train_id, never val/test."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from check_cnn import seed_all, pick_subset, describe_subset, evaluate_fixed, train_step
from dataset import FallDataset
from models.cnn_lstm import OriginalCNNLSTM, SenseFiStyleCNNLSTM
from models.sensefi_style import collate_device_segments

ROOT = Path(__file__).resolve().parent


def shapes(x):
    if isinstance(x, torch.Tensor):
        return list(x.shape)
    return [shapes(v) for v in x]


def save(path, value):
    path.write_text(json.dumps(value, indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-overfit-steps', type=int, default=500)
    args=ap.parse_args()
    if args.output_dir.exists() or args.max_overfit_steps<1:
        raise ValueError('Use a fresh output directory and positive max-overfit-steps')
    args.output_dir.mkdir(parents=True)
    paths=[ROOT/n for n in ('models/cnn.py','models/sensefi_style.py','dataset.py','preprocess.py',
        'outputs/preprocess_v1.json','outputs/cnn_seed42_mps_v1/best.pt',
        'outputs/cnn_seed42_mps_v1/frozen.pt','outputs/cnn_seed42_mps_v1/frozen.json',
        'outputs/sensefi_style_seed42_mps_v1/best.pt')]
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    torch.set_num_threads(2)
    dataset=FallDataset(ROOT/'reports/sample_manifest.csv',ROOT/'outputs/preprocess_v1.json','train_id')
    selected={n:pick_subset(dataset,k,args.seed) for n,k in [('shape',1),('smoke',16),('overfit',4)]}
    report={'seed':args.seed,'compute_device':'cpu','torch_version':str(torch.__version__),
        'split':'train_id','validation_or_test_used':False,'formal_training':False,
        'pretrained_weights':None,'optimizer':'Adam','learning_rate':.001,
        'loss':'BCEWithLogitsLoss, pos_weight=1 because diagnostic subsets are balanced',
        'threshold':.5,'subsets':{n:describe_subset(dataset,idx) for n,idx in selected.items()},'models':{}}
    loss_fn=nn.BCEWithLogitsLoss()
    start=time.monotonic()
    for name,cls,collate in [('original_cnn_lstm',OriginalCNNLSTM,None),
                            ('sensefi_style_cnn_lstm',SenseFiStyleCNNLSTM,collate_device_segments)]:
        seed_all(args.seed)
        model=cls(); trace=[];handles=[]
        for layer_name,layer in model.named_modules():
            if layer_name and not list(layer.children()):
                handles.append(layer.register_forward_hook(lambda m,i,o,n=layer_name:trace.append(
                    {'name':n,'layer':repr(m),'input':shapes(i[0]),'output':shapes(o)})))
        shape_batch=next(iter(DataLoader(Subset(dataset,selected['shape']),batch_size=4,collate_fn=collate)))
        model.eval()
        with torch.no_grad():
            out=model(shape_batch['x']);assert out.shape==(4,1) and torch.isfinite(out).all()
        for h in handles:h.remove()
        # Stateless calls and independent-device batches must agree in eval mode.
        with torch.no_grad():
            torch.testing.assert_close(out,model(shape_batch['x']),atol=1e-6,rtol=1e-5)
            if collate:
                offset=0
                for d in ('HP','ESP32'):
                    x=shape_batch['x'][d]
                    torch.testing.assert_close(model({d:x}),out[offset:offset+len(x)],atol=1e-6,rtol=1e-5)
                    offset+=len(x)
        r={'parameters':sum(p.numel() for p in model.parameters()),'layer_trace':trace,
           'shape_check_passed':True,'stateless_eval_check_passed':True}
        print(name,'SHAPE',json.dumps(r),flush=True)
        def fixed_batch(kind):
            return next(iter(DataLoader(Subset(dataset,selected[kind]),batch_size=len(selected[kind]),collate_fn=collate)))
        fixed=fixed_batch('smoke')
        initial=evaluate_fixed(model,fixed['x'],fixed['y'],loss_fn)
        initial_weights={n:p.detach().clone() for n,p in model.named_parameters()}
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        loader=DataLoader(Subset(dataset,selected['smoke']),batch_size=32,shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),collate_fn=collate)
        history=[]
        for epoch in (1,2):
            total=0.;grads=[]
            for batch in loader:
                loss,grad=train_step(model,optimizer,loss_fn,batch['x'],batch['y'])
                total+=loss*len(batch['y']);grads.append(grad)
            ev=evaluate_fixed(model,fixed['x'],fixed['y'],loss_fn)
            history.append({'epoch':epoch,'train_loss':total/64,'eval_loss_same_subset':ev['loss'],
                            'accuracy_same_subset':ev['accuracy'],'gradient_norms':grads})
            print(name,'SMOKE',json.dumps(history[-1]),flush=True)
        changes={n:float((p.detach()-initial_weights[n]).abs().max()) for n,p in model.named_parameters()}
        assert all(v>0 for v in changes.values())
        r['smoke']={'passed':True,'n':64,'epochs':2,'updates':4,'dropout':.3,'initial':initial,
                     'final':ev,'history':history,'max_parameter_changes':changes,
                     'sample_ids_in_score_order':fixed['sample_id'],'labels_in_score_order':fixed['y'].tolist()}
        seed_all(args.seed);model=cls(dropout=0.);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        fixed=fixed_batch('overfit');initial=evaluate_fixed(model,fixed['x'],fixed['y'],loss_fn)
        history=[{'step':0,'loss':initial['loss'],'accuracy':initial['accuracy']}]
        consecutive=0;passed=False
        for step in range(1,args.max_overfit_steps+1):
            loss,grad=train_step(model,optimizer,loss_fn,fixed['x'],fixed['y'])
            ev=evaluate_fixed(model,fixed['x'],fixed['y'],loss_fn)
            history.append({'step':step,'train_loss':loss,'gradient_norm':grad,'loss':ev['loss'],'accuracy':ev['accuracy']})
            consecutive=consecutive+1 if ev['correct']==16 and ev['loss']<.02 else 0
            if step==1 or step%25==0 or consecutive==5:
                print(name,'OVERFIT',json.dumps(history[-1]),flush=True)
            if consecutive==5:passed=True;break
        per_device={}
        for d in ('HP','ESP32'):
            idx=[i for i,v in enumerate(fixed['device']) if v==d]
            per_device[d]={'n':len(idx),'correct':sum((ev['scores'][i]>=.5)==bool(fixed['y'][i]) for i in idx)}
        r['overfit']={'passed':passed,'n':16,'dropout':0.,'steps':step,'max_steps':args.max_overfit_steps,
            'criterion':'all 16 correct and BCE<0.02 for 5 consecutive post-update eval checks',
            'initial':initial,'final':ev,'history':history,'per_device':per_device,
            'sample_ids_in_score_order':fixed['sample_id'],'labels_in_score_order':fixed['y'].tolist()}
        report['models'][name]=r
        save(args.output_dir/'diagnostics.json',report)
        with (args.output_dir/(name+'_overfit.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['step','train_loss','gradient_norm','loss','accuracy']);w.writeheader();w.writerows(history)
    report['original_artifacts_unchanged']=before=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    assert report['original_artifacts_unchanged']
    report['preserved_artifacts_sha256']=before;report['elapsed_seconds']=time.monotonic()-start
    save(args.output_dir/'diagnostics.json',report)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for name,r in report['models'].items():
        h=r['overfit']['history'];axes[0].plot([v['step'] for v in h],[v['loss'] for v in h],label=name)
        axes[1].plot([v['step'] for v in h],[v['accuracy'] for v in h],label=name)
    axes[0].set_ylabel('BCE on same 16 training segments');axes[1].set_ylabel('Accuracy on same 16 training segments')
    for ax in axes:ax.set_xlabel('Optimizer update');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('CNN+LSTM trainability checks only; no validation/test performance')
    fig.savefig(args.output_dir/'overfit_curves.png',dpi=150);plt.close(fig)
    print('DONE',json.dumps({n:{'smoke_passed':r['smoke']['passed'],'overfit_passed':r['overfit']['passed'],
        'steps':r['overfit']['steps'],'loss':r['overfit']['final']['loss']} for n,r in report['models'].items()}),flush=True)
    if not all(r['overfit']['passed'] for r in report['models'].values()):raise SystemExit(2)

if __name__=='__main__':main()
