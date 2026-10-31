
import json
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from night_common import NIGHT,atomic,stage_lock,log,now
from multi_common import SIZES,validate
from opt_v2_common import OUT2,window
from opt_v2_infer import infer_v2
from night_infer import infer_v3

def obj(k,x,y):return dict(kind=k,x=x,y=y,size=SIZES[k].tolist())
def main():
    if not (NIGHT/'candidate_frozen.json').exists() or not (NIGHT/'results.json').exists():raise RuntimeError('Wait for frozen final evaluation first')
    with stage_lock('custom_diagnostics'):
        from multi_simulate import build,place,channel
        b=joblib.load(NIGHT/'model.joblib');old=joblib.load(OUT2/'model.joblib');scene,objects,solver=build();rng=np.random.default_rng(2026103011);rows=[]
        cases=[('default_two',[obj(0,1,1),obj(2,5,5)]),('default_three',[obj(0,1,1),obj(1,3,3),obj(2,5,5)]),('user_current',[obj(0,3.84,3.42),obj(2,5,5)]),('previous_small_east',[obj(0,4.65,2.74),obj(2,5,5)]),('boundary_west',[obj(0,3.99,3.42),obj(2,5,5)]),('boundary_east',[obj(0,4.01,3.42),obj(2,5,5)])]
        with threadpool_limits(limits=4):
            for name,ps in cases:
                ps=validate(ps);place(objects,ps);h=channel(scene,solver);truth=sorted(o['zone'] for o in ps);result=[]
                for _ in range(32):
                    signal=window(h,rng,30);result.append(dict(v2=infer_v2(old,signal),v3=infer_v3(b,signal)))
                metrics={}
                for version in ['v2','v3']:
                    predictions=[sorted(r[version]['regions'][i]['zone'] for i in range(len(r[version]['regions']))) for r in result]
                    correct=np.array([p==truth for p in predictions]);accepted=np.array([r[version]['status']=='Accepted regions' for r in result]);layouts={}
                    for p in predictions:
                        k='+'.join('Z'+str(z+1) for z in p);layouts[k]=layouts.get(k,0)+1
                    metrics[version]=dict(correct_windows=int(correct.sum()),windows=32,exact_fraction=float(correct.mean()),accepted_windows=int(accepted.sum()),wrong_accepted=int(np.sum(~correct&accepted)),layout_histogram=layouts)
                rows.append(dict(name=name,objects=ps,true_center_zones=truth,metrics=metrics));log('Custom diagnostic '+name+': '+json.dumps(metrics))
        atomic('custom_diagnostics.json',dict(completed_at=now(),cases=rows,scope='Six predefined development/user examples;32 independent receiver-noise windows per one geometry, SNR30dB. Not an independent1500-scene test, no post-test tuning.'))

if __name__=='__main__':main()
