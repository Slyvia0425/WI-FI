
from pathlib import Path
import json,time,hashlib
import numpy as np,joblib
from threadpoolctl import threadpool_limits
from home_v5_physics import build,channel
from multi_simulate import place
from multi_common import SIZES,TX
from layout_registry import NEW_TX
from home_v5_window import window
from spatial_v6_infer import infer_v5,zone_prob,ordinal_post
from night_features import views
from home_v5_infer import batch as old_batch
from layout_t1_v1_common import STATE_COUNTS
O=Path('outputs/failure_diagnosis_20261031');O.mkdir(parents=True,exist_ok=True)
original=json.loads(Path('outputs/accuracy_retest_20261031/user_scene.json').read_text());ps=[dict(kind=o['kind'],x=o['x'],y=o['y'],size=(SIZES[o['kind']]*o['scale']).tolist()) for o in original['objects']]
views_config={'default':TX,'t1':NEW_TX,'extra_bottom_center':[list(v) for v in TX]};views_config['extra_bottom_center'][0]=[3.,.15,.8]

model=joblib.load('outputs/spatial_v6/default/model.joblib');ref=model['reference'];rms=np.sqrt(np.mean(np.abs(ref)**2,1,keepdims=True))
scene,objects,solver=build();channels={};rows=[]
cases={'all':ps,'without_small':[p for p in ps if p['kind']!=0],'without_medium':[p for p in ps if p['kind']!=1],'without_large':[p for p in ps if p['kind']!=2],'small_alone':[p for p in ps if p['kind']==0],'medium_alone':[p for p in ps if p['kind']==1],'large_alone':[p for p in ps if p['kind']==2]}
for lid,tx in views_config.items():
 for i,p in enumerate(tx):scene.transmitters['tx'+str(i)].position=p
 for name,placements in cases.items():
  place(objects,placements);h=channel(scene,solver);channels[lid+'_'+name]=h
  if lid=='default':
   truth=np.bincount([min(2,int(p['y']//2))*3+min(2,int(p['x']//2)) for p in placements],minlength=9).tolist();predictions=[]
   for seed in range(6000,6010):
    d=infer_v5(model,window(h,np.random.default_rng(seed),30));predictions.append(dict(seed=seed,pred=d.get('per_zone_object_count'),correct=d.get('per_zone_object_count')==truth,status=d['status'],joint=d.get('joint_confidence')))
   rows.append(dict(case=name,truth=truth,correct=sum(r['correct'] for r in predictions),n=10,predictions=predictions))

 place(objects,ps);channels[lid+'_all_repeat']=channel(scene,solver)
 print('Counterfactual view complete',lid,flush=True)
summary={}
for lid in views_config:
 h=channels[lid+'_all'];rep=channels[lid+'_all_repeat'];effects={}
 for kind,name in [(0,'small'),(1,'medium'),(2,'large')]:
  diff=channels[lid+'_all']-channels[lid+'_without_'+name];normalized=diff/np.maximum(rms,1e-10);perlink=np.sqrt(np.mean(np.abs(normalized)**2,1));pooled=normalized.reshape(32,8,8).mean(2)
  retained=float(np.sum(np.abs(pooled)**2)*8/max(np.sum(np.abs(normalized)**2),1e-30));noise_rms=np.sqrt(np.mean(np.abs(h)**2,1))/np.maximum(rms[:,0],1e-10)/np.sqrt(1000*16)
  effects[name]=dict(normalized_global_rms=float(np.sqrt(np.mean(np.abs(normalized)**2))),bin_average_retained_energy=retained,peak_link_to_nominal_averaged_noise_ratio=float(np.max(perlink/np.maximum(noise_rms,1e-10))),median_link_to_nominal_averaged_noise_ratio=float(np.median(perlink/np.maximum(noise_rms,1e-10))),top_links=np.argsort(perlink)[-6:][::-1].tolist(),per_link_effect=perlink.tolist())
 summary[lid]=dict(repeated_trace_relative_difference=float(np.linalg.norm(h-rep)/np.linalg.norm(h)),repeated_trace_max_abs_difference=float(np.max(np.abs(h-rep))),effects=effects)

h=window(channels['default_all'],np.random.default_rng(6000),30);v=views(h,ref);X=np.append(v['calibrated'],0)[None];nn=model['ordinal_refiner']['neural'];pn=nn['model'].predict_proba(nn['scaler'].transform(X));pb=model['ordinal_refiner']['boost'].predict_proba(X,thread_count=2)
head=dict(neural_zone_count_probabilities=zone_prob(pn)[0].tolist(),boost_zone_count_probabilities=zone_prob(pb)[0].tolist())
np.savez_compressed(O/'counterfactual_channels.npz',**channels)
report=dict(at=time.time(),scene=ps,views=views_config,signal_diagnosis=summary,model_counterfactuals=rows,count_heads=head,note='Diagnostic interventions on the previously opened user scene. Removing an object changes joint propagation, not an isolated additive component. Retained energy refers only to the 8-bin averaged complex branch, not all engineered features. Nominal noise ratios approximate independent averaged AWGN and exclude phase/gain drift. No model parameters changed; no claimed independent accuracy improvement.')
(O/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({'models':[{k:r[k] for k in ['case','correct','n']} for r in rows],'signals':{lid:{'repeat':s['repeated_trace_relative_difference'],'effects':{k:{z:v[z] for z in ['normalized_global_rms','bin_average_retained_energy','median_link_to_nominal_averaged_noise_ratio','peak_link_to_nominal_averaged_noise_ratio']} for k,v in s['effects'].items()}} for lid,s in summary.items()}},indent=2))
