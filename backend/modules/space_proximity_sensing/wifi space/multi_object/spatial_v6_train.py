
from pathlib import Path
import os,json,time,copy,hashlib,sys
import numpy as np,joblib
from threadpoolctl import threadpool_limits
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from layout_t1_v1_common import STATE_COUNTS
from home_v5_infer import batch as default_batch
from layout_t1_v3_infer import batch as t1_batch
O=Path('outputs/spatial_v6');O.mkdir(exist_ok=True,parents=True)
def atomic(n,d):
 p=O/n;t=p.with_suffix('.partial');t.write_text(json.dumps(d,indent=2),encoding='utf-8');t.replace(p)
def load(lid,split):
 p=Path('outputs')/('home_v5' if lid=='default' else 'layout_t1_v1')/'cache'/split
 return {k:np.load(p/(k+'.npy'),mmap_mode='r') for k in ['calibrated','zone_count','object_count','quality','phase_free']}
def design(d,lid,stride=1):return np.column_stack([np.asarray(d['calibrated'][::stride]),np.full((len(d['calibrated'][::stride]),1),0 if lid=='default' else 4,dtype=np.float32)])
def zone_prob(p):
 p=np.clip(np.asarray(p).reshape(-1,9,3),1e-6,1-1e-6);p=np.minimum.accumulate(p,axis=2)
 q=np.stack([1-p[:,:,0],p[:,:,0]-p[:,:,1],p[:,:,1]-p[:,:,2],p[:,:,2]],2);return np.clip(q,1e-7,1)
def score(p,base,w):
 q=zone_prob(p);logq=np.zeros((len(p),219))
 for z in range(9):logq+=np.log(q[:,z,STATE_COUNTS[:,z]])
 logits=np.log(np.maximum(base,1e-30))+w*logq;logits-=logits.max(1,keepdims=True);pp=np.exp(logits);return pp/pp.sum(1,keepdims=True)
def metrics(pp,d):
 pred=STATE_COUNTS[pp.argmax(1)];truth=np.asarray(d['zone_count']);exact=np.all(pred==truth,1);shared=np.sum(truth>0,1)<truth.sum(1);n=truth.sum(1)
 groups={str(k)+'_'+str(s):float(exact[(n==k)&(shared==s)].mean()) for k,s in [(1,False),(2,False),(3,False),(2,True),(3,True)]}
 return dict(joint=float(exact.mean()),balanced=float(np.mean(list(groups.values()))),groups=groups,count=float(np.mean(pred.sum(1)==n)),single=groups['1_False'],three=float(exact[n==3].mean()),shared=float(exact[shared].mean()))
def main():
 if (O/'selection.json').exists():raise RuntimeError('Selection already recorded; new version needed')
 atomic('state.json',dict(state='loading_training',pid=os.getpid()))
 data={lid:{s:load(lid,s) for s in ['train','validation']} for lid in ['default','t1_v1']}
 X=np.concatenate([design(data[lid]['train'],lid,2) for lid in data]);Y=np.concatenate([(np.asarray(data[lid]['train']['zone_count'][::2])[:,:,None]>=np.arange(1,4)).reshape(-1,27) for lid in data]).astype(np.int8)

 trcounts=np.concatenate([np.asarray(data[lid]['train']['zone_count'][::2]) for lid in data]);weights=(1+.75*(trcounts.max(1)>1)+.5*(trcounts.sum(1)==3)).astype(np.float32)
 V={lid:design(data[lid]['validation'],lid) for lid in data};bases={lid:joblib.load('outputs/'+('home_v5' if lid=='default' else 'layout_t1_v3')+'/model.joblib') for lid in data}
 old={lid:(default_batch if lid=='default' else t1_batch)(bases[lid],data[lid]['validation'])['posterior'] for lid in data}
 np.savez_compressed(O/'validation_baseline.npz',**old)
 scaler=StandardScaler().fit(X);X=scaler.transform(X);V={k:scaler.transform(v) for k,v in V.items()}
 atomic('protocol.json',dict(seed=2026103161,training='40000 existing training geometries per preset, one noise observation per geometry; ordinal targets>=1,>=2,>=3 per zone',validation='Original2400 per preset only',excluded='All opened final tests, prior diagnostic retest and user placement excluded from fitting and selection',models=['ordinal_mlp','ordinal_catboost'],epochs=100,gate='Validation gain>=1pp, balanced/shared/three no regression, single no drop>0.5pp. Freeze before new calibration/final scene generation. Fresh paired test gain>=1pp with95% CI lower>0; count/zone no drop>0.5pp. Confidence separately calibrated on fresh calibration scenes. Single GPU worker only.'))
 if (O/'ordinal_mlp.joblib').exists():
  saved=joblib.load(O/'ordinal_mlp.joblib');best=saved['model'];scaler=saved['scaler'];bestepoch=json.loads((O/'neural_trace.json').read_text())['best_epoch'];start=time.time()
 else:
  m=MLPClassifier(hidden_layer_sizes=(512,256),alpha=.2,batch_size=256,learning_rate_init=.0006,random_state=3161)
  best=None;bestscore=-1;bestepoch=0;trace=[];start=time.time()
  for epoch in range(1,101):
   m.partial_fit(X,Y,classes=np.arange(27),sample_weight=weights)
   if epoch%5:continue
   pp={lid:m.predict_proba(V[lid]) for lid in data};ev={lid:metrics(score(pp[lid],old[lid],1),data[lid]['validation']) for lid in data};val=np.mean([e['balanced'] for e in ev.values()]);trace.append(dict(epoch=epoch,metrics=ev))
   if val>bestscore:bestscore=val;best=copy.deepcopy(m);bestepoch=epoch
   atomic('state.json',dict(state='training_neural',pid=os.getpid(),epoch=epoch,best_epoch=bestepoch,elapsed_s=time.time()-start,validation=ev));print('MLP',epoch,'balanced',val,flush=True)
   if epoch>=45 and epoch-bestepoch>=25:break
  joblib.dump(dict(model=best,scaler=scaler),O/'ordinal_mlp.joblib');atomic('neural_trace.json',dict(trace=trace,best_epoch=bestepoch))

 from catboost import CatBoostClassifier
 Xraw=np.concatenate([design(data[lid]['train'],lid,2) for lid in data]);Vraw=np.concatenate([design(data[lid]['validation'],lid) for lid in data]);Yv=np.concatenate([(np.asarray(data[lid]['validation']['zone_count'])[:,:,None]>=np.arange(1,4)).reshape(-1,27) for lid in data]).astype(np.int8)
 atomic('state.json',dict(state='training_catboost',pid=os.getpid()))
 cb=CatBoostClassifier(loss_function='MultiLogloss',iterations=1600,depth=7,learning_rate=.06,l2_leaf_reg=12,random_seed=3162,task_type='GPU',devices='0',gpu_ram_part=.65,thread_count=3,allow_writing_files=False,verbose=150)
 cb.fit(Xraw,Y,sample_weight=weights,eval_set=(Vraw,Yv),early_stopping_rounds=180,use_best_model=True);joblib.dump(cb,O/'ordinal_catboost.joblib')
 preds={lid:{'neural':best.predict_proba(V[lid]),'boost':cb.predict_proba(design(data[lid]['validation'],lid),thread_count=2)} for lid in data}
 selection={}
 for lid in data:
  d=data[lid]['validation'];baseline=metrics(old[lid],d);candidates=[]
  for mix in [0.,.5,1.]:
   p=mix*preds[lid]['boost']+(1-mix)*preds[lid]['neural']
   for w in [.125,.25,.5,1.,2.,4.]:
    met=metrics(score(p,old[lid],w),d);candidates.append(dict(boost_weight=mix,ordinal_weight=w,**met))
  eligible=[c for c in candidates if c['joint']>=baseline['joint']+.01 and c['balanced']>=baseline['balanced'] and c['shared']>=baseline['shared'] and c['three']>=baseline['three'] and c['single']>=baseline['single']-.005 and c['count']>=baseline['count']-.005]
  choice=max(eligible,key=lambda a:(a['balanced'],a['joint'])) if eligible else None
  selection[lid]=dict(baseline=baseline,choice=choice,candidates=candidates,accepted_for_fresh_evaluation=choice is not None)
 atomic('selection.json',selection);atomic('state.json',dict(state='validation_selected',pid=os.getpid(),elapsed_s=time.time()-start,selection={lid:dict(baseline=v['baseline'],choice=v['choice']) for lid,v in selection.items()}));print(json.dumps(selection,indent=2),flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=3):main()

