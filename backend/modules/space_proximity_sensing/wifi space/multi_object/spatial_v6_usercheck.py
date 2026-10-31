
from pathlib import Path
import json,hashlib,time
import numpy as np,joblib
from threadpoolctl import threadpool_limits
from multi_common import SIZES,TX
from home_v5_physics import build,channel
from multi_simulate import place
from home_v5_window import window
from home_v5_infer import infer_v5 as old_infer
from spatial_v6_infer import infer_v5
O=Path('outputs/spatial_v6')
def main():
 assert (O/'test_opened.json').exists(),'Keep prior diagnostic out of selection'
 original=json.loads(Path('outputs/accuracy_retest_20261031/user_scene.json').read_text());ps=[dict(kind=o['kind'],x=o['x'],y=o['y'],size=(SIZES[o['kind']]*o['scale']).tolist()) for o in original['objects']]
 b=joblib.load(O/'default/model.joblib');old=b['base_bundle'];scene,objects,solver=build();place(objects,ps);h=channel(scene,solver);digest=hashlib.sha256(h.tobytes()).hexdigest();np.savez_compressed(O/'default/user_diagnostic_channel.npz',h=h)
 truth=[0,0,2,1,0,0,0,0,0];rows=[]
 for snr in [30,20,10]:
  for seed in range(4000,4020):
   obs=window(h,np.random.default_rng(seed),snr);a=old_infer(old,obs);d=infer_v5(b,obs)
   rows.append(dict(snr_db=snr,seed=seed,truth=truth,baseline=a.get('per_zone_object_count'),candidate=d.get('per_zone_object_count'),old_correct=a.get('per_zone_object_count')==truth,new_correct=d.get('per_zone_object_count')==truth,old_status=a['status'],new_status=d['status'],expert_agreement=d.get('expert_agreement'),joint_confidence=d.get('joint_confidence')))
 summary={str(snr):dict(n=20,baseline_correct=sum(r['old_correct'] for r in rows if r['snr_db']==snr),candidate_correct=sum(r['new_correct'] for r in rows if r['snr_db']==snr),new_detected=sum(r['new_status']=='Accepted regions' for r in rows if r['snr_db']==snr),new_wrong_detected=sum(r['new_status']=='Accepted regions' and not r['new_correct'] for r in rows if r['snr_db']==snr)) for snr in [30,20,10]}
 previous=[json.loads(s) for s in Path('outputs/accuracy_retest_20261031/observations.jsonl').read_text().splitlines()];expected={r['signal_sha256'] for r in previous if r['layout']=='default' and r['group']=='user_exact'}
 result=dict(at=time.time(),signal_sha256=digest,matches_previous_clean_channel=digest in expected,summary=summary,rows=rows,scope='Previously opened user diagnostic; post-freeze only, not independent accuracy or model selection.')
 (O/'default/user_diagnostic.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
if __name__=='__main__':
 with threadpool_limits(limits=3):main()
