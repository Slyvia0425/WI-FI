
import sys,tempfile,json,time,argparse
from pathlib import Path
import numpy as np
from multi_common import ROOT,DATA,OUT,CONFIG,SIZES,TX,RX,validate,target
from simulate import build_scene,write_ply

def build():
    import sionna.rt as rt
    scene,old,solver=build_scene();old.position=[3,3,-10]
    for name in list(scene.transmitters)+list(scene.receivers):scene.remove(name)
    for kind,positions,cls in [('tx',TX,rt.Transmitter),('rx',RX,rt.Receiver)]:
        for i,p in enumerate(positions):scene.add(cls(f'{kind}{i}',position=p))
    folder=Path(tempfile.gettempdir())/'wifi_multi_object_v1';folder.mkdir(exist_ok=True);mesh=folder/'unit_cube.ply'
    if not mesh.exists():write_ply(mesh,[-.5,-.5,-.5],[.5,.5,.5])
    material=rt.ITURadioMaterial('multi_wood','wood',thickness=.1)
    objects=[rt.SceneObject(fname=str(mesh),name=f'multi_{i}',radio_material=material) for i in range(3)]
    scene.edit(add=objects)
    for o in objects:o.position=[3,3,-10]
    return scene,objects,solver

def place(objects,placements):
    bykind={p['kind']:p for p in placements}
    for k,obj in enumerate(objects):
        if k in bykind:
            p=bykind[k];obj.scaling=p['size'];obj.position=[p['x'],p['y'],p['size'][2]/2]
        else:obj.position=[3,3,-10-2*k]

def channel(scene,solver):
    p=solver(scene,max_depth=3,samples_per_src=100000,seed=17,los=True,specular_reflection=True,diffuse_reflection=False,refraction=False)
    f=np.linspace(-10e6,10e6,64,endpoint=False,dtype=np.float32)
    h=np.asarray(p.cfr(frequencies=f,normalize_delays=False,normalize=False,out_type='numpy')).reshape(8,4,64).reshape(32,64)
    if not np.isfinite(h).all():raise ValueError('Nonfinite simulated channel')
    return h.astype(np.complex64)

def sample(rng,n):
    for _ in range(5000):
        kinds=rng.choice(3,size=n,replace=False);zones=rng.choice(9,size=n,replace=False);objects=[]
        for k,z in zip(kinds,zones):
            size=SIZES[k]*rng.uniform(.92,1.08,3);col=z%3;row=z//3
            low=[max(col*2,size[0]/2+.01),max(row*2,size[1]/2+.01)];high=[min(col*2+2,6-size[0]/2-.01),min(row*2+2,6-size[1]/2-.01)]
            x,y=rng.uniform(low,high);objects.append(dict(kind=int(k),x=float(x),y=float(y),size=size.tolist()))
        try:return validate(objects)
        except ValueError:pass
    raise RuntimeError('Cannot sample a valid multi-object scene')

def progress(**value):
    OUT.mkdir(exist_ok=True);p=OUT/'progress.json';tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)

def generate(pilot=False):
    DATA.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True);scene,objects,solver=build()
    empty=channel(scene,solver);np.save(DATA/'empty_reference.npy',empty)
    rng=np.random.default_rng(2026102923)
    if pilot:
        for n in [1,2,3]:
            ps=sample(rng,n);t=time.perf_counter();place(objects,ps);h=channel(scene,solver);print('objects',n,'seconds',time.perf_counter()-t,'relative response change',np.linalg.norm(h-empty)/np.linalg.norm(empty),'shape',h.shape,flush=True)
        return
    protocol=dict(CONFIG,seed=2026102923,splits=dict(train=3000,validation=600,calibration=300,test=900),count_probabilities=[.2,.4,.4],learning_target='Three size-category heads: absent=0, zone labels=1..9. No target count, dimensions or positions supplied at inference.')
    p=DATA/'protocol.json'
    if p.exists() and json.loads(p.read_text())!=protocol:raise ValueError('Protocol changed: use a separate data directory')
    p.write_text(json.dumps(protocol,indent=2));rows=[]
    for split,n in protocol['splits'].items():
        for i in range(n):
            count=int(rng.choice([1,2,3],p=[.2,.4,.4]));ps=sample(rng,count)
            rows.append(dict(id=f'{split}_{i:05}',split=split,objects=ps,target=target(ps).tolist(),count=count))
    (DATA/'manifest.json').write_text(json.dumps(rows,indent=2));records=DATA/'records';records.mkdir(exist_ok=True);start=time.time()
    for i,r in enumerate(rows):
        p=records/(r['id']+'.npz')
        cached=False
        if p.exists():
            try:
                with np.load(p,allow_pickle=False) as a:cached=a['h'].shape==(32,64) and np.isfinite(a['h']).all()
            except (OSError,ValueError,KeyError):pass
        if not cached:
            place(objects,r['objects']);tmp=p.with_suffix('.partial')
            with tmp.open('wb') as f:np.savez_compressed(f,h=channel(scene,solver))
            tmp.replace(p)
        if (i+1)%100==0:progress(state='generating',done=i+1,total=len(rows),elapsed_s=time.time()-start);print(i+1,'/',len(rows),round(time.time()-start,1),'s',flush=True)
    progress(state='generated',done=len(rows),total=len(rows),elapsed_s=time.time()-start)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');generate(parser.parse_args().pilot)
