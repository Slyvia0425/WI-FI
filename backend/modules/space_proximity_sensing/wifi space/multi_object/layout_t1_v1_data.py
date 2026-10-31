
import os,time,gc,json,tempfile,itertools,argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from multi_common import CONFIG,SIZES,validate,target
from night_data import one
from layout_t1_v1_physics import check,build,channel,SPEC
from layout_t1_v1_common import D5,O5,atomic,read,lock,sha,now

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
            try:return check(ps)
            except ValueError:pass
    raise RuntimeError('Cannot sample valid shared-zone geometry')

def prepare():
    D5.mkdir(exist_ok=True,parents=True);(D5/'records').mkdir(exist_ok=True)
    protocol=dict(version='layout-t1-v1',layout_id='t1_v1',seed=2026103071,physics=SPEC,new_train=40000,reuse_training='None; all new physics and geometry signals.',eval_splits=dict(validation=2400,calibration=1200,test=2400,boundary_test=600),sampling='Train:20% single and20% each two-distinct/three-distinct/two-shared/three-shared. Within shared-three,40% all one zone. Evaluation10% single,22.5% each other group; shared-three20% all one zone.',label_definition='1–3 movable wood cuboids, one per size band;219 center-zone count vectors. Fixed cabinet is calibrated background, excluded from target count.',test_policy='Final tests sealed until frozen model; never tune on final tests.',frames=16,receiver_model='25–40dB SNR; small gain/phase/frequency-slope drift. Not a complete802.11 PHY model.',release_rule='All-input test zone Macro-F1>=.85 and physical-count accuracy>=.85; spatial joint>=.65, shared count>=.80. Keep v4 available regardless.')
    pp=D5/'protocol.json'
    if pp.exists():
        if read(pp)!=protocol:raise RuntimeError('Protocol mismatch; use new version')
        if (D5/'manifest.json').exists():return read(D5/'manifest.json')
    pp.write_text(json.dumps(protocol,indent=2),encoding='utf-8');rng=np.random.default_rng(protocol['seed']);rows=[]
    for split,total in [('train',40000),*protocol['eval_splits'].items()]:
        if split=='train':
            jobs=[]
            for n,shared in [(1,False),(2,False),(3,False),(2,True),(3,True)]:
                jobs.extend((n,shared,bool(shared and n==3 and i%5<2)) for i in range(total//5))
        else:
            a=round(total*.1);b=(total-a)//4;parts=[a,b,b,b,total-a-3*b];jobs=[]
            for (n,shared),num in zip([(1,False),(2,False),(3,False),(2,True),(3,True)],parts):
                jobs.extend((n,shared,bool(shared and n==3 and i%5==0)) for i in range(num))
        rng.shuffle(jobs)
        for i,(n,shared,all_shared) in enumerate(jobs):
            ps=sample(rng,n,shared,split=='boundary_test',all_shared);id=f't1_{split}_{i:05}';zones={p['zone'] for p in ps}
            rows.append(dict(id=id,group_id=id,split=split,count=n,region_count=len(zones),shared=len(zones)<n,objects=ps,target=target(ps).tolist()))
            if (i+1)%2000==0:print('Prepared',split,i+1,flush=True)
    p=D5/'manifest.json';tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(rows,indent=2),encoding='utf-8');tmp.replace(p)
    atomic('manifest_audit.json',dict(prepared_at=now(),records=len(rows),manifest_sha256=sha(p),split_counts={s:sum(r['split']==s for r in rows) for s in ['train',*protocol['eval_splits']]},same_zone_supported=True));return rows

def worker(rows):
    cache=Path(tempfile.gettempdir())/('wifi_shared_v4_'+str(os.getpid()));cache.mkdir(exist_ok=True);tempfile.tempdir=str(cache)
    from multi_simulate import place
    sim=None;uses=0
    for r in rows:
        path=D5/'records'/(r['id']+'.npz')
        if path.exists():continue
        if sim is None or uses>=200:sim=None;gc.collect();sim=build();uses=0
        scene,objects,solver=sim;place(objects,r['objects']);h=channel(scene,solver);uses+=1;tmp=path.with_suffix('.'+str(os.getpid())+'.partial')
        with tmp.open('wb') as f:np.savez_compressed(f,h=h)
        tmp.replace(path)
    return len(rows)

def main(workers,batch_size=1000):
    with lock('generation'):
        rows=prepare();missing=[r for r in rows if not (D5/'records'/(r['id']+'.npz')).exists()]
        if not missing:atomic('data_state.json',dict(state='completed',pid=os.getpid(),updated=now(),done=len(rows),total=len(rows)));return
        with lock('gpu_data'):
            start=time.time()
            reference=D5/'empty_reference.npy'
            if not reference.exists():
                scene,objects,solver=build();np.save(reference,channel(scene,solver));del scene,objects,solver;gc.collect()


            with ProcessPoolExecutor(max_workers=workers,max_tasks_per_child=1) as pool:
                futures=[pool.submit(worker,missing[i:i+batch_size]) for i in range(0,len(missing),batch_size)]
                while not all(f.done() for f in futures):
                    for f in futures:
                        if f.done():f.result()
                    done=len(list((D5/'records').glob('*.npz')));atomic('data_state.json',dict(state='generating',pid=os.getpid(),updated=now(),done=done,total=len(rows),workers=workers,batch_size=batch_size,elapsed_s=time.time()-start));time.sleep(5)
                for f in futures:f.result()
            atomic('data_state.json',dict(state='completed',pid=os.getpid(),updated=now(),done=len(rows),total=len(rows),elapsed_s=time.time()-start))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=1);ap.add_argument('--batch-size',type=int,default=1000);a=ap.parse_args()
    if a.workers<1 or a.batch_size<1:ap.error('workers and batch-size must be positive')
    try:main(a.workers,a.batch_size)
    except Exception as e:
        import traceback
        atomic('data_state.json',dict(state='failed',pid=os.getpid(),updated=now(),error=str(e),traceback=traceback.format_exc()));raise
