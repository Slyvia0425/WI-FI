
import json,time,itertools
import numpy as np
from multi_common import CONFIG,SIZES,validate
from multi_simulate import build,place,channel
from opt_v2_common import DATA2,OUT2

def progress(**d):
    OUT2.mkdir(parents=True,exist_ok=True);p=OUT2/'progress.json';t=p.with_suffix('.partial');t.write_text(json.dumps(d,indent=2));t.replace(p)

def sample_for(rng,n,zones,boundary=False):
    for attempt in range(10000):
        kinds=rng.choice(3,n,replace=False);ps=[]
        for k,z in zip(kinds,zones):
            size=SIZES[k]*rng.uniform(.92,1.08,3);c=z%3;r=z//3
            lo=np.array([max(c*2,size[0]/2+.01),max(r*2,size[1]/2+.01)]);hi=np.array([min(c*2+2,6-size[0]/2-.01),min(r*2+2,6-size[1]/2-.01)])
            xy=rng.uniform(lo,hi)
            if boundary and k==kinds[0]:
                edges=[]
                for axis,index in [(0,c),(1,r)]:
                    if index>0:edges.append((axis,index*2,1))
                    if index<2:edges.append((axis,index*2+2,-1))
                axis,v,sign=edges[rng.integers(len(edges))];xy[axis]=v+sign*rng.uniform(.005,.15)
            ps.append(dict(kind=int(k),x=float(xy[0]),y=float(xy[1]),size=size.tolist()))
        try:return validate(ps)
        except ValueError:pass
    raise RuntimeError('Cannot sample a valid layout')

def main():
    DATA2.mkdir(parents=True,exist_ok=True);OUT2.mkdir(parents=True,exist_ok=True)
    protocol=dict(version=2,seed=2026103002,physical=CONFIG,observation_frames=16,phase_assumption='Frames represent one static synchronized scene. The optimized feature set includes phase-invariant and calibrated complex-response components; arbitrary receiver clock phases are unsupported.',splits=dict(train=12000,validation=900,calibration=600,test=1500,boundary_test=450),train_count_weights=[.1,.35,.55],eval_count_weights=[.1,.45,.45],reuse='Only the original 3000 training scenes may be added. Original validation/calibration/test remain diagnostic.',selection='Models and structured decoder selected on new validation, emphasizing complete two-/three-object layouts; new calibration and test held separate.')
    p=DATA2/'protocol.json'
    if p.exists() and json.loads(p.read_text())!=protocol:raise RuntimeError('Changed protocol')
    p.write_text(json.dumps(protocol,indent=2));rng=np.random.default_rng(protocol['seed']);rows=[]
    combos={n:list(itertools.combinations(range(9),n)) for n in [1,2,3]}
    for split,size in protocol['splits'].items():
        weights=protocol['train_count_weights'] if split=='train' else protocol['eval_count_weights']
        counts=[round(size*weights[0]),round(size*weights[1])];counts.append(size-sum(counts));jobs=[]
        for n,total in zip([1,2,3],counts):
            order=rng.permutation(len(combos[n]))
            for i in range(total):jobs.append((n,combos[n][int(order[i%len(order)])]))
        rng.shuffle(jobs)
        for i,(n,zs) in enumerate(jobs):
            ps=sample_for(rng,n,zs,boundary=split=='boundary_test' or (split=='train' and i%5==0))
            rows.append(dict(id=f'v2_{split}_{i:05}',split=split,count=n,objects=ps,paired_group=None))
    (DATA2/'manifest.json').write_text(json.dumps(rows,indent=2));folder=DATA2/'records';folder.mkdir(exist_ok=True)
    sim=None;used=0;start=time.time()
    for i,row in enumerate(rows):
        p=folder/(row['id']+'.npz')
        if p.exists():
            with np.load(p,allow_pickle=False) as a:
                if a['h'].shape!=(32,64) or not np.isfinite(a['h']).all():raise RuntimeError('Invalid cached scene')
            continue

        if sim is None or used>=150:sim=build();used=0
        scene,objects,solver=sim;place(objects,row['objects']);h=channel(scene,solver);used+=1
        tmp=p.with_suffix('.partial')
        with tmp.open('wb') as f:np.savez_compressed(f,h=h)
        tmp.replace(p)
        if (i+1)%100==0:progress(state='generating',done=i+1,total=len(rows),split=row['split'],elapsed_s=time.time()-start);print(i+1,'/',len(rows),row['split'],round(time.time()-start,1),'s',flush=True)
    progress(state='generated',done=len(rows),total=len(rows),elapsed_s=time.time()-start)

if __name__=='__main__':main()
