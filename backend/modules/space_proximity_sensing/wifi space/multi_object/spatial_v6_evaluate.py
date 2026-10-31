
from pathlib import Path
import json,time,hashlib,os,copy,sys
import numpy as np,joblib
from scipy.optimize import minimize_scalar
from threadpoolctl import threadpool_limits
from home_v5_data import sample
from tx_simulation import check
from multi_common import TX,DATA,OUT
from layout_registry import NEW_TX
from layout_t1_v1_common import STATE_COUNTS,state_ids
from layout_t1_v1_evaluate import metrics
from spatial_v6_infer import batch,base_batch
from night_infer import posterior
from home_v5_window import window
from night_features import views
O=OUT/'spatial_v6';D=DATA/'spatial_v6'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.partial');t.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8');t.replace(p)
def state(**kw):save(O/'evaluation_state.json',dict(pid=os.getpid(),**kw))
def seed(s):return int.from_bytes(hashlib.sha256(s.encode()).digest()[:4],'little')
def manifest():
 p=D/'manifest.json'
 if p.exists():return read(p)
 rng=np.random.default_rng(2026103167);rows=[]
 for split,N in [('calibration',600),('test',1200),('boundary_test',300)]:
  jobs=[];single=N//10;left=N-single;parts=[single]+[left//4]*3+[left-3*(left//4)]
  for (n,shared),nrows in zip([(1,False),(2,False),(3,False),(2,True),(3,True)],parts):
   for j in range(nrows):jobs.append((n,shared,shared and n==3 and j%5==0))
  rng.shuffle(jobs)
  for i,(n,shared,all_shared) in enumerate(jobs):
   while True:
    ps=sample(rng,n,shared,split=='boundary_test',all_shared)
    try:check(ps,NEW_TX);break
    except ValueError:pass
   rows.append(dict(id=f'v6_{split}_{i:05}',split=split,objects=ps,group_id=f'v6_{split}_{i:05}'))
 save(p,rows);return rows

def generate(rows,lids):
 from home_v5_physics import build,channel
 from multi_simulate import place
 scene=objects=solver=None;uses=0
 for lid in lids:
  for i,r in enumerate(rows):
   p=D/lid/'records'/(r['id']+'.npz')
   if p.exists():continue
   if scene is None or uses>=500:
    import gc
    scene=objects=solver=None;gc.collect();scene,objects,solver=build();uses=0
   for j,xyz in enumerate(TX if lid=='default' else NEW_TX):scene.transmitters[f'tx{j}'].position=xyz
   place(objects,r['objects']);h=channel(scene,solver);uses+=1;p.parent.mkdir(parents=True,exist_ok=True)
   with p.with_suffix('.partial').open('wb') as f:np.savez_compressed(f,h=h)
   p.with_suffix('.partial').replace(p)
   if (i+1)%200==0:state(state='generating_fresh',layout=lid,done=i+1,total=len(rows));print('DATA',lid,i+1,flush=True)

def features(rows,lid,split,b):
 p=O/lid/('features_'+split+'.npz')
 if p.exists():
  with np.load(p) as a:return {k:a[k] for k in a.files}
 selected=[r for r in rows if r['split']==split];counts=np.array([np.bincount([o['zone'] for o in r['objects']],minlength=9) for r in selected]);d=dict(zone_count=counts,labels=(counts>0).astype(int),object_count=counts.sum(1),ids=np.array([r['id'] for r in selected]));arr={k:[] for k in ['calibrated','phase_free','quality']}
 for r in selected:
  with np.load(D/lid/'records'/(r['id']+'.npz')) as a:h=window(a['h'],np.random.default_rng(seed('v6:'+lid+':'+r['id'])))
  v=views(h,b['reference'])
  for k in ['calibrated','phase_free']:arr[k].append(v[k])
  arr['quality'].append([v[q] for q in ['phase_health','temporal_coherence','roughness_cal','roughness_free','calibrated_usable']])
 d.update({k:np.asarray(a,np.float32) for k,a in arr.items()});np.savez_compressed(p,**d);return d

def wilson(correct,n):
 if not n:return 0
 z=1.6448536269514722;p=correct/n;return (p+z*z/(2*n)-z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)
def main():
 sel=read(O/'selection.json');lids=[k for k,v in sel.items() if v['accepted_for_fresh_evaluation']]
 if not lids:state(state='validation_rejected',note='No deployment change; preserve old models');return
 if (O/'test_opened.json').exists():raise RuntimeError('New final already opened; cannot refit')
 shared=dict(neural=joblib.load(O/'ordinal_mlp.joblib'),boost=joblib.load(O/'ordinal_catboost.joblib'))
 for lid in lids:
  folder=O/lid;folder.mkdir(parents=True,exist_ok=True)
  basepath=OUT/('home_v5' if lid=='default' else 'layout_t1_v3')/'model.joblib';base=joblib.load(basepath)
  wrapper=dict(base_bundle=base,ordinal_refiner=shared,choice=sel[lid]['choice'],layout_id=lid,revision='spatial_count_v6',reference=base['reference'],frames=16,version=5,temperature=1.,joint_gate=1.,require_agreement=True)
  joblib.dump(wrapper,folder/'selected.joblib')
 save(O/'selection_frozen.json',dict(at=time.time(),selection_sha256=sha(O/'selection.json'),protocol_sha256=sha(O/'protocol.json'),models={lid:sha(O/lid/'selected.joblib') for lid in lids},source_sha256={n:sha(Path(n)) for n in ['spatial_v6_train.py','spatial_v6_infer.py','spatial_v6_evaluate.py']},test_used=False))
 rows=manifest();state(state='generating_fresh');generate(rows,lids)

 geometry=lambda r:json.dumps(sorted((p['kind'],p['x'],p['y'],*p['size']) for p in r['objects']))
 known=set()
 for name in ['home_v5','layout_t1_v1','layout_t1_v2','layout_t1_v3']:
  known.update(geometry(r) for r in read(DATA/name/'manifest.json'))
 diagnostic=read(OUT/'accuracy_retest_20261031/plan.json');known.update(geometry(r) for r in diagnostic['scenes'])
 assert not any(geometry(r) in known for r in rows)
 for lid in lids:
  folder=O/lid;b=joblib.load(folder/'selected.joblib');d=features(rows,lid,'calibration',b);initial=batch(b,d);target=state_ids(d['zone_count']);raw=initial['raw_posterior']
  b['temperature']=float(minimize_scalar(lambda t:-np.log(np.maximum(posterior(raw,t)[np.arange(len(target)),target],1e-20)).mean(),bounds=(.5,4),method='bounded').x)
  r=batch(b,d);correct=np.all(r['zone_counts']==d['zone_count'],1);pp=r['posterior'].max(1);choices=[]
  for gate in np.linspace(.35,.999,100):
   keep=r['quality']&(pp>=gate)&r['witness_agreement']
   if keep.sum()>=60 and wilson(correct[keep].sum(),keep.sum())>=.90:choices.append(dict(gate=float(gate),n=int(keep.sum()),accuracy=float(correct[keep].mean()),lower=float(wilson(correct[keep].sum(),keep.sum()))))
  best=max(choices,key=lambda c:c['n']) if choices else None;b['joint_gate']=best['gate'] if best else 1.000001
  b['calibration_report']=dict(target='95% one-sided Wilson lower>=90% on calibrated accepted subset, at least60 of600',selected=best,require_agreement=True)
  joblib.dump(b,folder/'model.joblib');save(folder/'frozen.json',dict(at=time.time(),model_sha256=sha(folder/'model.joblib'),selected_sha256=sha(folder/'selected.joblib'),manifest_sha256=sha(D/'manifest.json'),source_sha256={n:sha(Path(n)) for n in ['spatial_v6_infer.py','spatial_v6_evaluate.py','home_v5_window.py','home_v5_physics.py','night_features.py','night_infer.py','layout_t1_v3_infer.py','layout_t1_v1_infer.py','layout_t1_v1_common.py','layout_t1_v1_evaluate.py']},test_opened=False));save(folder/'calibration.json',b['calibration_report'])
 save(O/'test_opened.json',dict(at=time.time(),models={lid:sha(O/lid/'model.joblib') for lid in lids}))
 for lid in lids:
  folder=O/lid;b=joblib.load(folder/'model.joblib');report=dict(selected_model='Ordinal per-zone count classification v6',split_counts=dict(train=80000,validation=4800,calibration=600,test=1200,boundary_test=300),splits={},baseline_comparison={},comparison=[])
  for split in ['calibration','test','boundary_test']:
   d=features(rows,lid,split,b);r=batch(b,d);old=base_batch(b,d);report['splits'][split]=metrics(d,r);report['baseline_comparison'][split]=dict(previous_t1_model=metrics(d,old));np.savez_compressed(folder/('predictions_'+split+'.npz'),truth=d['zone_count'],pred=r['zone_counts'],baseline=old['zone_counts'],accepted=r['accepted'],posterior=r['posterior'])
   if split=='test':
    delta=np.all(d['zone_count']==r['zone_counts'],1).astype(int)-np.all(d['zone_count']==old['zone_counts'],1).astype(int);rng=np.random.default_rng(3169);ci=np.quantile([rng.choice(delta,len(delta),replace=True).mean() for _ in range(3000)],[.025,.975]).tolist();report['paired']=dict(gain=float(delta.mean()),ci95=ci,improved=int((delta>0).sum()),worsened=int((delta<0).sum()))
   print('EVAL',lid,split,report['splits'][split]['exact_joint_accuracy'],flush=True)
  m=report['splits']['test'];old=report['baseline_comparison']['test']['previous_t1_model'];shared=m['by_sharing']['shared'];ci=report['paired']['ci95']
  guards=dict(joint_gain=report['paired']['gain']>=.01,ci_positive=ci[0]>0,zone_f1=m['occupied_zone_macro_f1']>=max(.85,old['occupied_zone_macro_f1']-.005),count=m['count_accuracy']>=max(.85,old['count_accuracy']-.005),three=m['by_true_count']['3']['exact_joint_accuracy']>=old['by_true_count']['3']['exact_joint_accuracy'],shared=shared['exact_joint_accuracy']>=old['by_sharing']['shared']['exact_joint_accuracy'],accepted_coverage=m['accepted_layout_coverage']>=.15,accepted_accuracy=(m['accepted_layout_accuracy'] or 0)>=.90,calibration_met=b['calibration_report']['selected'] is not None,accepted_not_lower=(m['accepted_layout_accuracy'] or 0)>=(old['accepted_layout_accuracy'] or 0),false_confirmation_not_more=m['accepted_layout_coverage']*(1-(m['accepted_layout_accuracy'] or 0))<=old['accepted_layout_coverage']*(1-(old['accepted_layout_accuracy'] or 0)))
  report['release_check']=dict(eligible=all(guards.values()),checks=guards);save(folder/'results.json',report);print(lid,'RELEASE',report['release_check'],flush=True)
 save(O/'data_audit.json',dict(passed=True,geometry_overlap=0,scenes_per_layout=len(rows),layouts=lids,manifest_sha256=sha(D/'manifest.json')));state(state='evaluated',layouts=lids)
if __name__=='__main__':
 try:
  with threadpool_limits(limits=3):main()
 except Exception as e:
  import traceback
  state(state='failed',error=str(e),traceback=traceback.format_exc());raise
