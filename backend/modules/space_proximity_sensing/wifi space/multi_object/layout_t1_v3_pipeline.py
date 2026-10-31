
import os,json,time,hashlib,copy,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np,joblib
from scipy.optimize import minimize_scalar
from threadpoolctl import threadpool_limits
from layout_t1_v1_common import read,sha,labels,zone_counts,state_ids
from layout_t1_v1_data import sample
from layout_t1_v1_physics import SPEC
from layout_t1_v1_evaluate import metrics
from layout_t1_v1_infer import batch as baseline_batch
from layout_t1_v3_infer import batch,refined_posterior,infer_v5
from layout_t1_v1_window import window
from night_features import views
from night_infer import posterior
from night_cache import QUALITY
from multi_common import DATA,OUT
O=OUT/'layout_t1_v3';D=DATA/'layout_t1_v3'
def atomic(name,obj):
 p=O/name;p.parent.mkdir(exist_ok=True,parents=True);t=p.with_name(p.name+'.'+str(os.getpid())+'.partial');t.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
 for attempt in range(40):
  try:t.replace(p);return
  except PermissionError:
   if attempt==39:raise
   time.sleep(.05)
def seed(s):return int.from_bytes(hashlib.sha256(s.encode()).digest()[:4],'little')
def worker(rows):
 import layout_t1_v1_data as data
 data.D5=D
 return data.worker(rows)
def prepare():
 D.mkdir(exist_ok=True);(D/'records').mkdir(exist_ok=True)
 p=D/'manifest.json'
 if p.exists():return read(p)
 rng=np.random.default_rng(2026103098);rows=[]
 for split,total in [('calibration',1200),('test',2400),('boundary_test',600)]:
  jobs=[]
  for (n,shared),num in zip([(1,False),(2,False),(3,False),(2,True),(3,True)],[total//10,*([int(total*.225)]*4)]):
   jobs.extend((n,shared,shared and n==3 and i%5==0) for i in range(num))
  rng.shuffle(jobs)
  for i,(n,shared,all_shared) in enumerate(jobs):
   ps=sample(rng,n,shared,split=='boundary_test',all_shared);id=f't3_{split}_{i:05}'
   rows.append(dict(id=id,group_id=id,split=split,count=n,region_count=len({p['zone'] for p in ps}),shared=len({p['zone'] for p in ps})<n,objects=ps))
 p.write_text(json.dumps(rows,indent=2),encoding='utf-8')
 atomic('protocol.json',dict(seed=2026103098,scenes=4200,physics=SPEC,reused_training='V1 plus new zone-count regressors trained on original40000 physical scenes; selection uses original validation only',fresh_splits=dict(calibration=1200,test=2400,boundary_test=600),gate='Paired fresh test: joint gain>=1pp,95% paired bootstrap CI lower>0, three-object joint not lower, zone/count no drop>0.5pp; original absolute release criteria also required. No test-driven refitting.',gpu_workers=1,batch_size=1000))
 return rows
def features(rows,split,b):
 selected=[r for r in rows if r['split']==split];p=O/('features_'+split+'.npz')
 if p.exists():
  with np.load(p) as a:return {k:a[k] for k in a.files}
 yy,cc=labels(selected);data=dict(labels=yy,object_count=cc,zone_count=zone_counts(selected),ids=np.array([r['id'] for r in selected]));arr={k:[] for k in ['calibrated','phase_free','quality']}
 for r in selected:
  with np.load(D/'records'/(r['id']+'.npz')) as a:h=window(a['h'],np.random.default_rng(seed('t3-window:'+r['id'])))
  v=views(h,b['reference'])
  for k in ['calibrated','phase_free']:arr[k].append(v[k])
  arr['quality'].append([v[q] for q in QUALITY])
 data.update({k:np.array(v,np.float32) for k,v in arr.items()});np.savez_compressed(p,**data);return data
def audit(rows):
 old=read(DATA/'layout_t1_v1/manifest.json')+read(DATA/'layout_t1_v2/manifest.json');geometry=lambda r:json.dumps(sorted((p['kind'],p['x'],p['y'],*p['size']) for p in r['objects']))
 known={geometry(r) for r in old};assert not any(geometry(r) in known for r in rows)
 oldhash=set()
 for r in old:
  with np.load(DATA/('layout_t1_v2' if r['id'].startswith('t2_') else 'layout_t1_v1')/'records'/(r['id']+'.npz')) as a:oldhash.add(hashlib.sha256(a['h'].tobytes()).hexdigest())
 seen={};overlap=0
 for r in rows:
  with np.load(D/'records'/(r['id']+'.npz')) as a:h=a['h']
  assert h.shape==(32,64) and np.isfinite(h).all() and np.linalg.norm(h)>0
  digest=hashlib.sha256(h.tobytes()).hexdigest();overlap+=digest in oldhash
  assert digest not in seen or seen[digest]==r['split'],'New split overlap'
  seen[digest]=r['split']
 assert overlap==0
 atomic('data_audit.json',dict(passed=True,new_scenes=len(rows),old_scenes_checked=len(old),geometry_overlap=0,channel_overlap=0))
def main():
 O.mkdir(exist_ok=True);sel=read(O/'selection.json');assert sel['accepted_for_fresh_evaluation']
 assert sel['best']['regressor'] in ['boost','forest','neural','mixed','boost_neural','forest_neural']
 assert sha(OUT/'layout_t1_v1/model.joblib')==sel['base_model_sha256']
 atomic('state.json',dict(state='waiting_for_training',pid=os.getpid()))
 for kind in ['boost','forest','neural']:
  while not (O/(kind+'.joblib')).exists():
   state=read(O/(kind+'_state.json'));assert state['state']!='failed',state
   time.sleep(3)
 from layout_t1_v3_train import select
 if not (O/'frozen.json').exists():select()
 sel=read(O/'selection.json');assert sel['accepted_for_fresh_evaluation']
 rows=prepare();missing=[r for r in rows if not (D/'records'/(r['id']+'.npz')).exists()]
 with ProcessPoolExecutor(max_workers=1,max_tasks_per_child=1) as pool:
  jobs=[pool.submit(worker,missing[i:i+1000]) for i in range(0,len(missing),1000)]
  while not all(j.done() for j in jobs):
   for j in jobs:
    if j.done():j.result()
   atomic('state.json',dict(state='generating_fresh_evaluation',pid=os.getpid(),done=len(list((D/'records').glob('*.npz'))),total=len(rows)));time.sleep(5)
  for j in jobs:j.result()
 atomic('state.json',dict(state='auditing',pid=os.getpid()));audit(rows)
 base=joblib.load(OUT/'layout_t1_v1/model.joblib');b=copy.deepcopy(base)
 if (O/'frozen.json').exists():b=joblib.load(O/'model.joblib');assert sha(O/'model.joblib')==read(O/'frozen.json')['model_sha256']
 else:
  choice=sel['best'];name=choice['regressor'];names=['boost','forest'] if name=='mixed' else name.split('_')
  b.update(revision='t1_count_v3',count_refiner=dict(models=[dict(name=n,model=joblib.load(O/(n+'.joblib'))) for n in names],scale=choice['scale'],weight=choice['weight']),refiner_temperature=1.)
  d=features(rows,'calibration',b);ids=state_ids(d['zone_count']);post=refined_posterior(b,d)
  b['refiner_temperature']=float(minimize_scalar(lambda t:-np.log(np.maximum(posterior(post,t)[np.arange(len(ids)),ids],1e-20)).mean(),bounds=(.5,4),method='bounded').x)
  r=batch(b,d);correct=np.all(r['zone_counts']==d['zone_count'],axis=1);choices=[]
  for gate in np.linspace(.35,.99,65):
   k=r['quality']&(r['posterior'].max(1)>=gate)
   if k.sum()>=40 and correct[k].mean()>=.9:choices.append(float(gate))
  b['joint_gate']=min(choices) if choices else .99;joblib.dump(b,O/'model.joblib')
  atomic('frozen.json',dict(model_sha256=sha(O/'model.joblib'),base_model_sha256=sha(OUT/'layout_t1_v1/model.joblib'),selection_sha256=sha(O/'selection.json'),source_sha256={n:sha(Path(n)) for n in ['layout_t1_v3_infer.py','layout_t1_v3_pipeline.py','layout_t1_v1_infer.py','layout_t1_v1_physics.py','home_v5_physics.py','tx_simulation.py','layout_t1_v1_common.py','night_features.py']},frozen_at=time.time(),test_used=False))
 if not (O/'test_opened.json').exists():atomic('test_opened.json',dict(at=time.time(),model_sha256=sha(O/'model.joblib')))
 report=dict(selected_model='Per-zone physical count refinement v3',model_bytes=(O/'model.joblib').stat().st_size,split_counts=dict(train=40000,validation=2400,calibration=1200,test=2400,boundary_test=600),splits={},baseline_comparison={},comparison=[],scope='Paired entirely new4200 scenes; no fitting on final test. V1 training components reused.')
 paired=None
 for split in ['calibration','test','boundary_test']:
  d=features(rows,split,b);r=batch(b,d);old=baseline_batch(base,d);report['splits'][split]=metrics(d,r);report['baseline_comparison'][split]=dict(previous_t1_model=metrics(d,old))
  if split=='test':paired=np.all(r['zone_counts']==d['zone_count'],axis=1).astype(int)-np.all(old['zone_counts']==d['zone_count'],axis=1).astype(int)
  np.savez_compressed(O/('predictions_'+split+'.npz'),ids=d['ids'],truth=d['zone_count'],pred=r['zone_counts'],baseline=old['zone_counts'])
 m=report['splits']['test'];old=report['baseline_comparison']['test']['previous_t1_model'];rng=np.random.default_rng(3099);ci=np.quantile([rng.choice(paired,len(paired),replace=True).mean() for _ in range(3000)],[.025,.975]).tolist();gain=float(paired.mean())
 checks=dict(zone_f1_85=m['occupied_zone_macro_f1']>=.85,physical_count_85=m['count_accuracy']>=.85,full_spatial_count_65=m['exact_joint_accuracy']>=.65,shared_physical_count_80=m['by_sharing']['shared']['count_accuracy']>=.8,paired_gain=gain>=.01,paired_ci_positive=ci[0]>0,no_zone_drop=m['occupied_zone_macro_f1']>=old['occupied_zone_macro_f1']-.005,no_count_drop=m['count_accuracy']>=old['count_accuracy']-.005,three_not_lower=m['by_true_count']['3']['exact_joint_accuracy']>=old['by_true_count']['3']['exact_joint_accuracy'],single_guard=m['by_true_count']['1']['exact_joint_accuracy']>=old['by_true_count']['1']['exact_joint_accuracy']-.005)
 report['paired_comparison']=dict(joint_gain=gain,ci95=ci,improved_scenes=int((paired>0).sum()),worsened_scenes=int((paired<0).sum()));report['release_check']=dict(eligible=all(checks.values()),checks=checks)
 atomic('results.json',report);atomic('state.json',dict(state='evaluated',release=report['release_check'],paired=report['paired_comparison']));print(json.dumps(dict(metrics={k:m[k] for k in ['occupied_zone_macro_f1','count_accuracy','exact_joint_accuracy']},paired=report['paired_comparison'],release=report['release_check']),indent=2))
if __name__=='__main__':
 try:
  with threadpool_limits(limits=4):main()
 except Exception as e:atomic('state.json',dict(state='failed',error=str(e),traceback=traceback.format_exc()));raise
