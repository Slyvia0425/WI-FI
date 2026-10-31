
import os, json, time, argparse, hashlib, tempfile, shutil
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
DATA = ROOT/'data'

def check_generation_config():
    DATA.mkdir(parents=True,exist_ok=True)
    path=DATA/'generation_config.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=CONFIG:
        raise RuntimeError('Config differs from cached data. Use a separate experiment/data directory; do not mix generated signals.')
    if not path.exists():path.write_text(json.dumps(CONFIG,ensure_ascii=False,indent=2),encoding='utf-8')

def write_ply(path, lo, hi):
    x,y,z=lo; X,Y,Z=hi
    v=[(x,y,z),(X,y,z),(X,Y,z),(x,Y,z),(x,y,Z),(X,y,Z),(X,Y,Z),(x,Y,Z)]
    faces=[(0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),(1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7)]
    s='ply\nformat ascii 1.0\nelement vertex 8\nproperty float x\nproperty float y\nproperty float z\nelement face 12\nproperty list uchar int vertex_indices\nend_header\n'
    s+=''.join(' '.join(map(str,p))+'\n' for p in v)
    s+=''.join('3 '+' '.join(map(str,p))+'\n' for p in faces)
    path.write_text(s,encoding='ascii')

def build_scene(material_shift=False,device_shift=False):
    import sionna.rt as rt
    import mitsuba as mi
    scene_dir=ROOT/'scene';scene_dir.mkdir(parents=True,exist_ok=True)


    asset_dir=Path(tempfile.gettempdir())/'wifi_static_scene_v1';asset_dir.mkdir(exist_ok=True)
    specs={'floor':([-0.1,-0.1,-0.15],[6.1,6.1,0]),
           'ceiling':([-0.1,-0.1,3],[6.1,6.1,3.15]),
           'west':([-0.15,0,0],[0,6,3]),'east':([6,0,0],[6.15,6,3]),
           'south':([-0.15,-0.15,0],[6.15,0,3]),'north':([-0.15,6,0],[6.15,6.15,3]),
           'obstacle':([-0.6,-0.4,-0.6],[0.6,0.4,0.6])}
    xml='<scene version="3.0.0"><bsdf type="diffuse" id="itu_concrete"><rgb name="reflectance" value="0.5,0.5,0.5"/></bsdf><bsdf type="diffuse" id="itu_wood"><rgb name="reflectance" value="0.5,0.3,0.2"/></bsdf>'
    for name,(lo,hi) in specs.items():
        mesh=scene_dir/(name+'.ply')
        if not mesh.exists():write_ply(mesh,lo,hi)
        material='itu_wood' if name=='obstacle' else 'itu_concrete'
        cached=asset_dir/mesh.name;shutil.copyfile(mesh,cached)
        xml+=f'<shape type="ply" id="{name}"><string name="filename" value="{cached.as_posix()}"/><ref id="{material}"/></shape>'
    xml+='</scene>'
    xmlpath=scene_dir/'room.xml';xmlpath.write_text(xml,encoding='utf-8')

    scene=rt.load_scene_from_string(xml,merge_shapes=False)
    scene.frequency=CONFIG['carrier_hz']
    scene.tx_array=rt.PlanarArray(num_rows=1,num_cols=1,vertical_spacing=.5,horizontal_spacing=.5,pattern='iso',polarization='V')
    scene.rx_array=rt.PlanarArray(num_rows=1,num_cols=1,vertical_spacing=.5,horizontal_spacing=.5,pattern='iso',polarization='V')
    for kind,positions in [('tx',CONFIG['tx_m']),('rx',CONFIG['rx_m'])]:
        for i,p in enumerate(positions):
            p=np.array(p,dtype=float)
            if device_shift:p[:2]+=np.array([.06,-.04])*(1 if i%2==0 else -1)
            scene.add((rt.Transmitter if kind=='tx' else rt.Receiver)(f'{kind}{i}',position=p.tolist()))
    obj=scene.get('obstacle')
    if material_shift:
        obj.radio_material=rt.ITURadioMaterial('shifted_object','chipboard',thickness=.1)
    return scene,obj,rt.PathSolver()

def response(scene,solver,seed):
    paths=solver(scene,max_depth=CONFIG['max_depth'],samples_per_src=CONFIG['samples_per_src'],seed=int(seed),los=True,specular_reflection=True,diffuse_reflection=False,refraction=False)
    frequencies=np.linspace(-CONFIG['bandwidth_hz']/2,CONFIG['bandwidth_hz']/2,CONFIG['subcarriers'],endpoint=False,dtype=np.float32)
    cfr=paths.cfr(frequencies=frequencies,normalize_delays=False,normalize=False,out_type='numpy')
    h=(np.asarray(cfr[0])+1j*np.asarray(cfr[1])) if isinstance(cfr,tuple) else np.asarray(cfr)

    return h.reshape(len(CONFIG['rx_m']),len(CONFIG['tx_m']),CONFIG['subcarriers']).reshape(-1,CONFIG['subcarriers']).astype(np.complex64)

def status(**kw):
    DATA.mkdir(parents=True,exist_ok=True)
    (DATA/'progress.json').write_text(json.dumps(kw,ensure_ascii=False,indent=2),encoding='utf-8')

def generate(pilot=False):
    if not pilot and (DATA/'manifest_phase1.json').exists():raise RuntimeError('Phase-2 dataset already exists. Use robust_generate.py/rebuild.ps1; do not overwrite the manifest.')
    check_generation_config()
    import sionna.rt
    import mitsuba as mi
    DATA.mkdir(parents=True,exist_ok=True)
    print('Sionna loaded; Mitsuba variant:',mi.variant(),flush=True)
    scene,obj,solver=build_scene();obj.position=[3,3,-10]
    t=time.perf_counter();ref=response(scene,solver,17)
    print('Reference',ref.shape,'finite',np.isfinite(ref).all(),'seconds',time.perf_counter()-t,flush=True)
    np.save(DATA/'empty_reference.npy',ref)
    if pilot:
        hs=[]
        for i in range(9):
            obj.position=[1+2*(i%3),1+2*(i//3),.6]
            t=time.perf_counter();h=response(scene,solver,17);hs.append(h)
            print('Zone',i+1,'rel change',float(np.linalg.norm(h-ref)/np.linalg.norm(ref)),'seconds',time.perf_counter()-t,flush=True)
        np.savez_compressed(DATA/'pilot.npz',h=np.array(hs),reference=ref)
        return
    rng=np.random.default_rng(CONFIG['seed'])
    jobs=[]
    for split,n in CONFIG['split_positions_per_zone'].items():
        for zone in range(9):
            for index in range(n):
                center=np.array([1+2*(zone%3),1+2*(zone//3)])
                xy=center+rng.uniform(-np.array(CONFIG['position_jitter_m']),np.array(CONFIG['position_jitter_m']))
                jobs.append(dict(id=f'{split}_Z{zone+1}_{index:03}',split=split,zone=zone,xy=xy.tolist(),seed=17))
    (DATA/'manifest.json').write_text(json.dumps(jobs,indent=2),encoding='utf-8')
    active='';started=time.time();done=0
    for job in jobs:
        split=job['split'];dest=DATA/'records'/(job['id']+'.npz');dest.parent.mkdir(exist_ok=True)
        if dest.exists():
            with np.load(dest,allow_pickle=False) as p:
                if p['h'].shape!=ref.shape or not np.isfinite(p['h']).all():raise ValueError('Invalid cached record '+str(dest))
            done+=1;continue
        if active!=split:
            scene,obj,solver=build_scene(material_shift=split=='material_shift',device_shift=split=='device_shift');active=split
        obj.position=[*job['xy'],.6]
        t=time.perf_counter();h=response(scene,solver,job['seed'])
        if not np.isfinite(h).all():raise ValueError('Nonfinite RT result')
        np.savez_compressed(dest,h=h)
        done+=1
        status(state='generating',done=done,total=len(jobs),split=split,elapsed_s=time.time()-started,last_ms=(time.perf_counter()-t)*1000)
        if done%10==0:print(done,'/',len(jobs),split,'elapsed',round(time.time()-started,1),flush=True)
    status(state='generated',done=done,total=len(jobs),elapsed_s=time.time()-started,engine='Sionna RT '+sionna.rt.__version__,variant=mi.variant())

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
    generate(args.pilot)
