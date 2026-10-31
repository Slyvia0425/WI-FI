
import os,json,time,gc,tempfile,argparse,itertools,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from multi_common import CONFIG,SIZES,DATA,validate,target
from night_common import NIGHT,atomic,stage_lock,alive,log,now,DEADLINE,TZ
from datetime import datetime
D3=DATA/'overnight_v3'

def size_for(rng,k):return SIZES[k]*rng.uniform(.92,1.08,3)
def bounds(z,size):
    c=z%3;r=z//3
    return np.array([max(2*c,size[0]/2+.01),max(2*r,size[1]/2+.01)]),np.array([min(2*c+2,6-size[0]/2-.01),min(2*r+2,6-size[1]/2-.01)])
def one(rng,k,z,boundary=False):
    size=size_for(rng,k);lo,hi=bounds(z,size);xy=rng.uniform(lo,hi)
    if boundary:
        edges=[]
        for axis,index in [(0,z%3),(1,z//3)]:
            if index>0:edges.append((axis,index*2,1))
            if index<2:edges.append((axis,index*2+2,-1))
        axis,v,sign=edges[rng.integers(len(edges))];xy[axis]=v+sign*rng.uniform(.005,.20)
    return dict(kind=int(k),x=float(xy[0]),y=float(xy[1]),size=size.tolist())
def layout(rng,kinds,zones,boundary=False):
    for _ in range(10000):
        ps=[one(rng,k,z,boundary=boundary and i==0) for i,(k,z) in enumerate(zip(kinds,zones))]
        try:return validate(ps)
        except ValueError:pass
    raise RuntimeError('No valid layout')
def add_one(rng,base,k,zone):
    for _ in range(2000):
        obj=one(rng,k,zone)
        try:return validate(base+[obj])
        except ValueError:pass
    raise RuntimeError('No valid paired addition')
def paired_variants(rng,base,k):
    occupied={p['zone'] for p in base};free=[z for z in range(9) if z not in occupied]
    edges=[(a,b) for a,b in itertools.combinations(free,2) if (a//3==b//3 and b-a==1) or b-a==3]
    if edges and rng.random()<.65:
        for _ in range(250):
            a,b=edges[rng.integers(len(edges))];size=size_for(rng,k);loa,hia=bounds(a,size);lob,hib=bounds(b,size)
            axis=0 if b-a==1 else 1;other=1-axis;edge=2*(b%3 if axis==0 else b//3)
            value=rng.uniform(max(loa[other],lob[other]),min(hia[other],hib[other]));p1=np.zeros(2);p2=np.zeros(2);p1[other]=p2[other]=value;p1[axis]=edge-rng.uniform(.005,.20);p2[axis]=edge+rng.uniform(.005,.20)
            objs=[dict(kind=int(k),x=float(p[0]),y=float(p[1]),size=size.tolist()) for p in [p1,p2]]
            try:return validate(base+[objs[0]]),validate(base+[objs[1]]),'cross_boundary'
            except ValueError:pass

    for _ in range(30):
        zs=rng.choice(free,2,replace=False)
        try:return add_one(rng,base,k,int(zs[0])),add_one(rng,base,k,int(zs[1])),'different_zones'
        except RuntimeError:pass
    raise RuntimeError('Cannot create paired variants')

def prepare():
    D3.mkdir(exist_ok=True,parents=True);(D3/'records').mkdir(exist_ok=True)
    protocol=dict(version='overnight-v3',seed=2026103003,physical=CONFIG,uniform_training=12000,paired_training_groups=3000,records_per_pair_group=3,eval_splits=dict(validation=1200,calibration=600,test=1500,boundary_test=450),train_count_weights=[.1,.35,.55],eval_count_weights=[.1,.45,.45],paired_missing_kind_weights=[.7,.15,.15],paired_boundary_probability=.65,boundary_distance_m=[.005,.20],reuse='Original 3000 train and v2 12000 train only. All old validation/calibration/test excluded.',group_policy='Each counterfactual background group stays entirely in training. Evaluation backgrounds are independent.',test_policy='Generate and integrity-audit final test, but never score it before candidate is frozen.',observation_frames=16)
    pp=D3/'protocol.json'
    if pp.exists():
        if json.loads(pp.read_text())!=protocol:raise RuntimeError('Night protocol changed; use a new directory')
        if (D3/'manifest.json').exists():return json.loads((D3/'manifest.json').read_text())
    pp.write_text(json.dumps(protocol,indent=2));rng=np.random.default_rng(protocol['seed']);rows=[]
    def row(id,split,objects,group,mode):return dict(id=id,split=split,count=len(objects),objects=objects,target=target(objects).tolist(),group_id=group,mode=mode)
    combos={n:list(itertools.combinations(range(9),n)) for n in [1,2,3]}
    for split,total in [('train',protocol['uniform_training']),*protocol['eval_splits'].items()]:
        weights=protocol['train_count_weights'] if split=='train' else protocol['eval_count_weights'];counts=[round(total*weights[0]),round(total*weights[1])];counts.append(total-sum(counts));jobs=[]
        for n,count in zip([1,2,3],counts):
            order=rng.permutation(len(combos[n]));jobs.extend((n,combos[n][int(order[j%len(order)])]) for j in range(count))
        rng.shuffle(jobs)
        for i,(n,zs) in enumerate(jobs):
            kinds=rng.choice(3,n,replace=False);ps=layout(rng,kinds,zs,boundary=split=='boundary_test' or (split=='train' and i%5==0));id=f'v3_{split}_uniform_{i:05}';rows.append(row(id,split,ps,id,'uniform' if split!='boundary_test' else 'boundary'))
    for i in range(protocol['paired_training_groups']):
        missing=int(rng.choice(3,p=protocol['paired_missing_kind_weights']));kinds=[k for k in range(3) if k!=missing];zs=rng.choice(9,2,replace=False);base=layout(rng,kinds,zs);a,b,mode=paired_variants(rng,base,missing);group=f'v3_pair_group_{i:05}'
        for suffix,ps in [('base',base),('add_a',a),('add_b',b)]:rows.append(row(f'v3_train_pair_{i:05}_{suffix}','train',ps,group,'pair_'+mode+'_'+suffix))
        if (i+1)%500==0:print('Prepared paired groups',i+1,flush=True)

    order={'train':0,'validation':1,'calibration':2,'test':3,'boundary_test':4};rows.sort(key=lambda r:order[r['split']])
    manifest=D3/'manifest.json';tmp=manifest.with_suffix('.partial');tmp.write_text(json.dumps(rows,indent=2));tmp.replace(manifest)
    sha=hashlib.sha256(manifest.read_bytes()).hexdigest();atomic('data_manifest.json',dict(state='prepared',at=now(),sha256=sha,records=len(rows),split_counts={s:sum(r['split']==s for r in rows) for s in order},training_pair_groups=protocol['paired_training_groups']));log('Fresh v3 manifest prepared: '+str(len(rows))+' physical scenes, paired backgrounds training-only.');return rows

def worker(rows):
    cache=Path(tempfile.gettempdir())/('wifi_night_v3_'+str(os.getpid()));cache.mkdir(exist_ok=True);tempfile.tempdir=str(cache)
    from multi_simulate import build,place,channel
    sim=None;used=0;made=0;previous_group=None
    for row in rows:
        if datetime.now(TZ)>=DEADLINE:break
        p=D3/'records'/(row['id']+'.npz')
        if p.exists():continue
        if sim is None or (used>=100 and previous_group!=row['group_id']):sim=None;gc.collect();sim=build();used=0
        scene,objects,solver=sim;place(objects,row['objects']);h=channel(scene,solver);used+=1
        previous_group=row['group_id']
        tmp=p.with_suffix('.'+str(os.getpid())+'.partial')
        with tmp.open('wb') as f:np.savez_compressed(f,h=h)
        tmp.replace(p);made+=1
    return made

def generate(rows,workers=3):
    with stage_lock('data_generation'):
        lock=NIGHT/'catboost_pilot.lock'
        while lock.exists():
            try:busy=alive(json.loads(lock.read_text())['pid'])
            except (OSError,ValueError,KeyError):busy=False
            if not busy:break
            atomic('data_state.json',dict(state='waiting_for_pilot_gpu',pid=os.getpid(),updated=now()));time.sleep(5)
        with stage_lock('gpu_dataset'):
            missing=[r for r in rows if not (D3/'records'/(r['id']+'.npz')).exists()];before=len(rows)-len(missing);start=time.time();log('v3 joint ray tracing starts/resumes: '+str(before)+' cached, '+str(len(missing))+' remaining.')


            buckets=[[] for _ in range(workers)];rng=np.random.default_rng(2026103017)
            for split in ['train','validation','calibration','test','boundary_test']:
                groups={}
                for row in missing:
                    if row['split']==split:groups.setdefault(row['group_id'],[]).append(row)
                keys=list(groups);rng.shuffle(keys)
                for key in keys:buckets[int(np.argmin([len(b) for b in buckets]))].extend(groups[key])
            with ProcessPoolExecutor(max_workers=workers) as pool:
                fs=[pool.submit(worker,bucket) for bucket in buckets]
                while not all(f.done() for f in fs):
                    for f in fs:
                        if f.done():f.result()
                    done=len(list((D3/'records').glob('*.npz')));atomic('data_state.json',dict(state='generating',pid=os.getpid(),updated=now(),done=done,total=len(rows),workers=workers,elapsed_s=time.time()-start));time.sleep(5)
                for f in fs:f.result()
            done=len(list((D3/'records').glob('*.npz')));atomic('data_state.json',dict(state='completed' if done==len(rows) else 'deadline_stopped',pid=os.getpid(),updated=now(),done=done,total=len(rows),elapsed_s=time.time()-start));log('v3 ray tracing finished/stopped with '+str(done)+'/'+str(len(rows))+' physical scenes.')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--workers',type=int,default=3);args=ap.parse_args()
    try:
        rows=prepare()
        if not args.prepare_only:generate(rows,args.workers)
    except Exception as e:
        import traceback
        atomic('data_state.json',dict(state='failed',pid=os.getpid(),updated=now(),error=str(e),traceback=traceback.format_exc()));raise
