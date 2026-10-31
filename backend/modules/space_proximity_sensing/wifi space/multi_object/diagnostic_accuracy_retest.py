
from pathlib import Path
import json,time,collections,hashlib,sys
import numpy as np
from home_v5_verify import request
from multi_common import TX,SIZES
from layout_registry import NEW_TX
from tx_simulation import check as check_physical
from home_v5_data import sample
ROOT=Path(__file__).resolve().parent
O=ROOT/'outputs'/'accuracy_retest_20261031';O.mkdir(parents=True,exist_ok=True)
if (O/'observations.jsonl').exists():raise FileExistsError('Retest evidence exists; choose a new output directory for a new run.')
source=Path(r'C:/Users/xiuye/Documents/Codex/2026-10-04/https-github-com-abinand2631-wifi-vision/outputs/accuracy_retest_20261031/user_scene.json')
original=json.loads(source.read_text(encoding='utf-8'));(O/'user_scene.json').write_text(json.dumps(original,indent=2),encoding='utf-8')
models=[ROOT/'outputs'/n/'model.joblib' for n in ['home_v5','layout_t1_v3']]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
model_shas={str(p.relative_to(ROOT)):sha(p) for p in models}
base='http://127.0.0.1:8769';code,state=request(base,'/api/state');assert code==200
user=[dict(kind=o['kind'],x=o['x'],y=o['y'],size=(SIZES[o['kind']]*o['scale']).tolist()) for o in original['objects']]
scenes=[dict(id='user_exact',group='user_exact',objects=user)]
rng=np.random.default_rng(202610315530)

while len(scenes)<9:
 ps=[dict(o,x=round(o['x']+rng.uniform(-.08,.08),3),y=round(o['y']+rng.uniform(-.08,.08),3)) for o in user]
 try:
  for tx in [TX,NEW_TX]:check_physical(ps,tx)
 except ValueError:continue
 scenes.append(dict(id='nearby_'+str(len(scenes)),group='nearby',objects=ps))
for n,shared in [(1,False),(2,False),(3,False),(2,True),(3,True)]:
 for j in range(6):
  while True:
   ps=sample(rng,n,shared,all_shared=shared and n==3 and j<2)
   try:check_physical(ps,NEW_TX);break
   except ValueError:pass
  scenes.append(dict(id=f'fresh_{n}_{shared}_{j}',group=f'fresh_{n}_'+('shared' if shared else 'distinct'),objects=ps))
for j in range(6):
 while True:
  ps=sample(rng,3,j%2==0,boundary=True)
  try:check_physical(ps,NEW_TX);break
  except ValueError:pass
 scenes.append(dict(id=f'boundary_{j}',group='fresh_boundary',objects=ps))
plan=dict(seed=202610315530,created_at=time.time(),scenes=scenes,user_repeat_seeds=list(range(4000,4020)),user_snrs=[30,20,10],fresh_repeats=3,nearby_repeats=5,replay_per_layout=120,note='Diagnostic sampling fixed before prediction. Repeated noise windows are correlated and not independent scenes. Different radio presets produce different CSI; no same-input algorithm comparison is claimed. No retraining or threshold changes.')
(O/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
rows=[];errors=[]
def capture(lid,group,scene_id,snr,seed,path,body):
 start=time.perf_counter();code,d=request(base,path,body)
 if code!=200:
  errors.append(dict(layout=lid,scene=scene_id,code=code,error=d));return
 o=d['occupancy'];truth=[0]*9
 for p in d['truth']:truth[p['zone']]+=1
 pred=o.get('per_zone_object_count');exact=pred==truth
 row=dict(layout=lid,revision=d['inference_model_revision'],group=group,scene_id=scene_id,snr_db=snr,seed=seed,truth=truth,predicted=pred,status=o['status'],joint_confidence=o.get('joint_confidence'),count_correct=o.get('estimated_object_count')==sum(truth),occupied_zones_correct=pred is not None and [v>0 for v in pred]==[v>0 for v in truth],exact=exact,probabilities=o.get('probabilities'),alternatives=o.get('alternative_layouts'),signal_sha256=d.get('signal_sha256'),roundtrip_ms=(time.perf_counter()-start)*1000)
 rows.append(row)
 with (O/'observations.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
for lid,tx in [('default',TX),('t1_v1',NEW_TX)]:
 for i,scene in enumerate(scenes):
  repeats=20 if scene['group']=='user_exact' else 5 if scene['group']=='nearby' else 3
  for snr in ([30,20,10] if scene['group']=='user_exact' else [30]):
   for repeat in range(repeats):
    seed=4000+repeat;capture(lid,scene['group'],scene['id'],snr,seed,'/api/simulate',dict(objects=scene['objects'],tx_positions=tx,seed=seed,snr_db=snr))
  if i%10==0:print(lid,'custom scenes',i+1,'observations',len(rows),flush=True)

 available=[r for r in state['records'] if r['layout_id']==lid and r['split']=='test']
 for n,shared,num in [(1,False,24),(2,False,24),(3,False,24),(2,True,24),(3,True,24)]:
  pool=[r for r in available if r['count']==n and bool(r['shared'])==shared]
  selected=np.random.default_rng(202610315531).choice(len(pool),num,replace=False)
  for k in selected:
   r=pool[int(k)];capture(lid,'replay_'+str(n)+'_'+('shared' if shared else 'distinct'),r['id'],30,4230,'/api/record',dict(id=r['id'],layout_id=lid,seed=4230,snr_db=30))
 print(lid,'completed',len(rows),flush=True)
def stats(rs):
 n=len(rs);accepted=[r for r in rs if r['status']=='Accepted regions'];wrong=[r for r in rs if not r['exact']]
 return dict(n=n,distinct_scenes=len({r['scene_id'] for r in rs}),exact_correct=sum(r['exact'] for r in rs),exact_accuracy=sum(r['exact'] for r in rs)/n if n else None,count_accuracy=sum(r['count_correct'] for r in rs)/n if n else None,occupied_zone_accuracy=sum(r['occupied_zones_correct'] for r in rs)/n if n else None,statuses=dict(collections.Counter(r['status'] for r in rs)),accepted_n=len(accepted),accepted_wrong=sum(not r['exact'] for r in accepted),wrong_n=len(wrong),wrong_predictions=dict(collections.Counter(str(r['predicted']) for r in wrong)))
summary={}
for lid in ['default','t1_v1']:
 rr=[r for r in rows if r['layout']==lid]
 summary[lid]=dict(user_by_snr={str(snr):stats([r for r in rr if r['group']=='user_exact' and r['snr_db']==snr]) for snr in [30,20,10]},nearby=stats([r for r in rr if r['group']=='nearby']),fresh_all=stats([r for r in rr if r['group'].startswith('fresh_')]),replay_all=stats([r for r in rr if r['group'].startswith('replay_')]),by_group={g:stats([r for r in rr if r['group']==g]) for g in sorted({r['group'] for r in rr if r['group']!='user_exact'})})
assert {str(p.relative_to(ROOT)):sha(p) for p in models}==model_shas
result=dict(completed=True,total_observations=len(rows),errors=errors,summary=summary,models_unchanged=True,model_sha256=model_shas,note=plan['note'])
(O/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({lid:{'user30':v['user_by_snr']['30'],'fresh':v['fresh_all'],'replay':v['replay_all']} for lid,v in summary.items()},indent=2),flush=True)
