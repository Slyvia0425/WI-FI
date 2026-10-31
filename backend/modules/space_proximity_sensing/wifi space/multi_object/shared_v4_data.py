
import os,time,gc,json,tempfile,itertools,argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from multi_common import CONFIG,SIZES,validate,target
from night_data import one
from shared_v4_common import D4,O4,atomic,read,lock,sha,now

def sample(rng,n,shared,boundary=False,all_shared=False):
    for trial in range(10000):
        kinds=rng.choice(3,n,replace=False)
        if shared:
            z=int(rng.integers(9));zones=[z]*n
            if n==3 and not all_shared:
                zones[2]=int(rng.choice([v for v in range(9) if v!=z]))
            rng.shuffle(zones)
        else:zones=rng.choice(9,n,replace=False)

        for _ in range(25 if shared and n==3 else 5):
            ps=[one(rng,k,z,boundary=boundary and i==0) for i,(k,z) in enumerate(zip(kinds,zones))]
            try:return validate(ps,allow_shared_zones=True)
            except ValueError:pass
    raise RuntimeError('Cannot sample valid shared-zone geometry')

def prepare():
    D4.mkdir(exist_ok=True,parents=True);(D4/'records').mkdir(exist_ok=True)
    protocol=dict(version='shared-zone-v4',seed=2026103014,physics=dict(CONFIG,notes='Shared center zones allowed; physical non-overlap and radio clearance retained.'),new_shared_train=12000,reuse_training='All36000 original/v2/v3 TRAIN scenes only; no previous held-out scenes.',eval_splits=dict(validation=1200,calibration=800,test=1800,boundary_test=450),sampling='Shared train:45% two objects/one zone,45% three/two zones,10% three/one zone. Evaluation:10% single,22.5% each two-distinct/three-distinct/two-shared/three-shared. Within shared-three,20% use one zone.',label_definition='Center occupancy9 binary; occupied-zone count1..3; physical-object count1..3;219 per-zone multiplicity states. Distinct size bands and physical non-overlap/device clearance still apply.',test_policy='Generate/audit raw data, but do not score final test before frozen.json. Background groups independent across splits.',frames=16)
    pp=D4/'protocol.json'
    if pp.exists():
        if read(pp)!=protocol:raise RuntimeError('Protocol mismatch; use new version')
        if (D4/'manifest.json').exists():return read(D4/'manifest.json')
    pp.write_text(json.dumps(protocol,indent=2),encoding='utf-8');rng=np.random.default_rng(protocol['seed']);rows=[]
    for split,total in [('train',12000),*protocol['eval_splits'].items()]:
        if split=='train':jobs=[(2,True,False)]*5400+[(3,True,False)]*5400+[(3,True,True)]*1200
        else:
            a=round(total*.1);b=(total-a)//4;parts=[a,b,b,b,total-a-3*b];jobs=[]
            for (n,s),num in zip([(1,False),(2,False),(3,False),(2,True),(3,True)],parts):
                jobs.extend((n,s,bool(s and n==3 and i%5==0)) for i in range(num))
        rng.shuffle(jobs)
        for i,(n,shared,all_shared) in enumerate(jobs):
            ps=sample(rng,n,shared,split=='boundary_test',all_shared);id=f's4_{split}_{i:05}';zones={p['zone'] for p in ps}
            rows.append(dict(id=id,group_id=id,split=split,count=n,region_count=len(zones),shared=len(zones)<n,objects=ps,target=target(ps).tolist()))
            if (i+1)%2000==0:print('Prepared',split,i+1,flush=True)
    p=D4/'manifest.json';tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(rows,indent=2),encoding='utf-8');tmp.replace(p)
    atomic('manifest_audit.json',dict(prepared_at=now(),records=len(rows),manifest_sha256=sha(p),split_counts={s:sum(r['split']==s for r in rows) for s in ['train',*protocol['eval_splits']]},same_zone_supported=True));return rows

def worker(rows):
    cache=Path(tempfile.gettempdir())/('wifi_shared_v4_'+str(os.getpid()));cache.mkdir(exist_ok=True);tempfile.tempdir=str(cache)
    from multi_simulate import build,place,channel
    sim=None;uses=0
    for r in rows:
        path=D4/'records'/(r['id']+'.npz')
        if path.exists():continue
        if sim is None or uses>=100:sim=None;gc.collect();sim=build();uses=0
        scene,objects,solver=sim;place(objects,r['objects']);h=channel(scene,solver);uses+=1;tmp=path.with_suffix('.'+str(os.getpid())+'.partial')
        with tmp.open('wb') as f:np.savez_compressed(f,h=h)
        tmp.replace(path)
    return len(rows)

def main(workers):
    with lock('generation'):
        rows=prepare();missing=[r for r in rows if not (D4/'records'/(r['id']+'.npz')).exists()]
        if not missing:atomic('data_state.json',dict(state='completed',pid=os.getpid(),updated=now(),done=len(rows),total=len(rows)));return
        with lock('gpu_data'):
            start=time.time()
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures=[pool.submit(worker,missing[i::workers]) for i in range(workers)]
                while not all(f.done() for f in futures):
                    for f in futures:
                        if f.done():f.result()
                    done=len(list((D4/'records').glob('*.npz')));atomic('data_state.json',dict(state='generating',pid=os.getpid(),updated=now(),done=done,total=len(rows),workers=workers,elapsed_s=time.time()-start));time.sleep(5)
                for f in futures:f.result()
            atomic('data_state.json',dict(state='completed',pid=os.getpid(),updated=now(),done=len(rows),total=len(rows),elapsed_s=time.time()-start))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);a=ap.parse_args()
    try:main(a.workers)
    except Exception as e:
        import traceback
        atomic('data_state.json',dict(state='failed',pid=os.getpid(),updated=now(),error=str(e),traceback=traceback.format_exc()));raise
