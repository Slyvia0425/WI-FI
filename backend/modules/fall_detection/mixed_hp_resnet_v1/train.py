"""Train a separate random-initialized HP mixed-source candidate, never the test set."""
import argparse,json,os,platform,time
from collections import Counter
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from common import *

def save_checkpoint(path,state):
 tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)

def plot_history(out,history):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 f,ax=plt.subplots(1,2,figsize=(11,4));epochs=[r['epoch'] for r in history]
 for key in ('train_loss','val_loss'):ax[0].plot(epochs,[r[key] for r in history],label=key)
 for key in ('precision','recall','F1'):ax[1].plot(epochs,[r['FallDetection_val_05'][key] for r in history],label=key)
 ax[0].set_title('Weighted cross entropy');ax[1].set_title('HP FallDetection validation, threshold 0.5')
 for a in ax:a.set_xlabel('Epoch');a.legend();a.grid(alpha=.2)
 f.tight_layout();f.savefig(out/'training_curves.png',dpi=150);plt.close(f)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--device',default='cuda');ap.add_argument('--batch-size',type=int,default=64);ap.add_argument('--resume',action='store_true');ap.add_argument('--max-epochs',type=int,default=100);a=ap.parse_args()
 torch.set_num_threads(2);seed_all(42);device=torch.device(a.device)
 train=CachedDataset(a.cache,'train');val=CachedDataset(a.cache,'val');verify_isolation(train.rows,val.rows)
 audit=json.loads(a.audit.read_text());assert len(train)==audit['train_n'] and len(val)==audit['val_n']
 assert all(sha(a.cache/(s+'.npy'))==audit['cache_sha256'][s] for s in ('train','val'))
 counts=Counter(r['label'] for r in train.rows);weights=torch.tensor([len(train)/(2*counts[i]) for i in range(2)],dtype=torch.float32,device=device)
 config={'model_config':MODEL_CONFIG,'label_mapping':LABELS,'seed':42,'batch_size':a.batch_size,'max_epochs':a.max_epochs,'early_stopping_patience':10,'optimizer':'AdamW','learning_rate':.001,'weight_decay':1e-5,'scheduler':None,'initialization':'random; no pretrained weights','loss':'CrossEntropyLoss; train-derived weights N/(2*N_class)','class_counts':dict(counts),'class_weights':weights.cpu().tolist(),'checkpoint_selection':'minimum combined-source weighted validation CE','preprocess_config':{'input_shape':[1,500,232],'raw_shape':[232,500,1],'h5_key':'CSI_amps','axes':'raw feature,time,link -> link,time,feature','normalization':'per-record scalar mean/std, correction1; std clamped1e-8','padding':False,'augmentation':False},'data_version':'CSI-Bench Kaggle v12','selection_sha256':audit['selection_sha256'],'train_content_hashes_sha256':audit['train_content_hashes_sha256'],'val_content_hashes_sha256':audit['val_content_hashes_sha256'],'cache_sha256':audit['cache_sha256'],'comparison_thresholds':[.5,OLD_THRESHOLD],'device':str(device),'hardware':torch.cuda.get_device_name() if device.type=='cuda' else platform.machine(),'torch':str(torch.__version__),'source_sha256':{n:sha(Path(__file__).parent/n) for n in ('common.py','train.py')}}
 a.output.mkdir(parents=True,exist_ok=True)
 protocol=a.output/'training_config.json'
 if protocol.exists():
  assert a.resume,'Output already exists: use explicit --resume'
  assert json.loads(protocol.read_text())==json.loads(json.dumps(config)),'Resume configuration changed'
 else:dump(protocol,config)
 model=ResNet18Classifier(**MODEL_CONFIG).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-5)
 criterion=torch.nn.CrossEntropyLoss(weight=weights);history=[];start=1;best=float('inf');bad=0;best_epoch=0
 if a.resume:
  old=torch.load(a.output/'last.pt',map_location='cpu',weights_only=True)
  assert old['config']==config
  model.load_state_dict(old['model_state_dict']);optimizer.load_state_dict(old['optimizer_state_dict']);history=old['history'];start=old['epoch']+1;best=old['best_val_loss'];bad=old['bad_epochs'];best_epoch=old['best_epoch'];torch.set_rng_state(old['torch_rng'])
  if device.type=='cuda':torch.cuda.set_rng_state_all(old['cuda_rng'])
 val_loader=torch.utils.data.DataLoader(val,batch_size=a.batch_size,shuffle=False,num_workers=0)
 print('FROZEN_PROTOCOL',json.dumps(config),flush=True)
 # A finite forward/backward check is part of epoch 1; there is no extra hidden fitting pass.
 stop='max_epochs'
 for epoch in range(start,a.max_epochs+1):
  if bad>=10:stop='early_stopping';break
  begin=time.time();model.train();num=0.;den=0.
  gen=torch.Generator().manual_seed(42+epoch)
  loader=torch.utils.data.DataLoader(train,batch_size=a.batch_size,shuffle=True,generator=gen,num_workers=0)
  for x,y,_ in loader:
   x=x.to(device);y=y.to(device);optimizer.zero_grad(set_to_none=True);out=model(x);loss=criterion(out,y)
   if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
   loss.backward()
   finite_grads=torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all()
   if not finite_grads:raise ValueError('Nonfinite gradient')
   optimizer.step();batch_weight=weights[y].sum().item();num+=loss.item()*batch_weight;den+=batch_weight
  scores,vl=predict(model,val_loader,device,weights)
  cr=[i for i,r in enumerate(val.rows) if r['source']=='FallDetection'];hr=[i for i,r in enumerate(val.rows) if r['source']=='HAR']
  row={'epoch':epoch,'train_loss':num/den,'val_loss':vl,'seconds':time.time()-begin,'FallDetection_val_05':metrics([val.rows[i] for i in cr],[scores[i] for i in cr],.5),'HAR_val_triggered_05':sum(scores[i]>=.5 for i in hr)}
  history.append(row)
  improved=vl<best
  if improved:best=vl;best_epoch=epoch;bad=0
  else:bad+=1
  state={'model_state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'model_config':MODEL_CONFIG,'preprocess_config':config['preprocess_config'],'normalization_parameters':'per-record statistic; none fitted on validation','label_mapping':LABELS,'seed':42,'best_epoch':best_epoch,'epoch':epoch,'best_val_loss':best,'bad_epochs':bad,'data_version':config['data_version'],'split_identifiers':{'train':[r['id'] for r in train.rows],'val':[r['id'] for r in val.rows]},'config':config,'model_version':'hp_mixed_resnet18_v1','history':history,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else []}
  if improved:
   save_checkpoint(a.output/'best.pt',state)
   dump(a.output/'best_reference.json',{'id':val.rows[0]['id'],'score':scores[0],'epoch':epoch,'checkpoint_sha256':sha(a.output/'best.pt')})
  state['optimizer_state_dict']=optimizer.state_dict();save_checkpoint(a.output/'last.pt',state)
  dump(a.output/'history.json',history);dump(a.output/'status.json',{'status':'running','epoch':epoch,'best_epoch':best_epoch,'bad_epochs':bad,'last_epoch':row});plot_history(a.output,history)
  print('EPOCH',json.dumps(row),'best',best_epoch,'patience',bad,flush=True)
 else:stop='max_epochs'
 if bad>=10:stop='early_stopping'
 dump(a.output/'status.json',{'status':'complete','stop_reason':stop,'epochs':len(history),'best_epoch':best_epoch,'best_val_loss':best,'checkpoint_sha256':sha(a.output/'best.pt')})
 print('TRAINING_COMPLETE',json.dumps(json.loads((a.output/'status.json').read_text())),flush=True)

if __name__=='__main__':main()
