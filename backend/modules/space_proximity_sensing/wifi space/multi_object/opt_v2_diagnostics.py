
import json,time,urllib.request
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from opt_v2_common import *
from opt_v2_infer import infer_v2
from occupancy import infer_occupancy
from opt_v2_train import metrics

def main():
    b=joblib.load(OUT2/'model.joblib');old=joblib.load(OUT/'occupancy.joblib');rows=json.loads((DATA2/'manifest.json').read_text());tests=[r for r in rows if r['split']=='test'][:120];stress={}
    with threadpool_limits(limits=4):
        for mode in ['30dB','20dB','10dB','phase_drift','single_frame']:
            ys=[];ps=[];unknown=0;accepted=0
            for i,r in enumerate(tests):
                with np.load(DATA2/'records'/(r['id']+'.npz')) as a:h=a['h']
                obs=window(h,np.random.default_rng(617+i),float(mode[:-2]) if mode.endswith('dB') else 30)
                if mode=='phase_drift':obs*=np.exp(1j*np.random.default_rng(17+i).uniform(-np.pi,np.pi,(1,32,1))).astype(np.complex64)
                if mode=='single_frame':obs=obs[0]
                d=infer_v2(b,obs);pred=np.zeros(9)
                for o in d['regions']:pred[o['zone']]=1
                ys.append(targets([r])[0]);ps.append(pred);unknown+=d['status']=='Unknown';accepted+=d['status']=='Accepted regions'
            stress[mode]=dict(metrics=metrics(np.array(ys),np.array(ps)),unknown=unknown,accepted=accepted,note='Supplemental fixed 120-scene diagnostic; not the official full test. Phase drift deliberately violates calibration assumptions.')
    scenes=[('default_two',[dict(kind=0,x=1,y=1),dict(kind=2,x=5,y=5)]),('default_three',[dict(kind=0,x=1,y=1),dict(kind=1,x=3,y=3),dict(kind=2,x=5,y=5)]),('user_visible_failure',[dict(kind=0,x=4.65,y=2.74),dict(kind=2,x=5,y=5)])];examples=[]
    from multi_simulate import build,place,channel
    from multi_common import validate
    scene,objects,solver=build()
    for name,ps in scenes:
        truth=validate(ps);place(objects,truth);h=channel(scene,solver);obs=window(h,np.random.default_rng(42),30);entry=dict(name=name,truth=truth,old_single=infer_occupancy(old,obs[0]),old_averaged=infer_occupancy(old,obs.mean(0)),optimized=infer_v2(b,obs));effects=[]
        for removed in truth:
            place(objects,[p for p in truth if p['kind']!=removed['kind']]);ablated=channel(scene,solver);effects.append(dict(removed_kind=removed['kind'],removed_zone=removed['zone'],relative_channel_change=float(np.linalg.norm(h-ablated)/np.linalg.norm(h))))
        entry['per_object_signal_effect']=effects;examples.append(entry)

    recompute=[]
    for n in [1,2,3]:
        r=next(r for r in rows if r['split']=='test' and r['count']==n);place(objects,r['objects']);fresh=channel(scene,solver)
        with np.load(DATA2/'records'/(r['id']+'.npz')) as a:stored=a['h']
        error=float(np.linalg.norm(fresh-stored)/np.linalg.norm(stored));recompute.append(dict(id=r['id'],count=n,relative_l2=error,passed=error<.005))
    (OUT2/'diagnostics.json').write_text(json.dumps(dict(stress=stress,known_scenes=examples,physical_recompute=recompute),indent=2,allow_nan=False));print('Diagnostics saved',flush=True)
if __name__=='__main__':main()
