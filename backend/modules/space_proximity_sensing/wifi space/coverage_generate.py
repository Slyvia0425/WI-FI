
import json,time
import numpy as np
from simulate import build_scene,response,check_generation_config,CONFIG
from coverage_common import ROOT,DATA,OUT,BOUNDS,CLEARANCE,valid_placement,near_boundary

def progress(**d):
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/'progress.json';q=p.with_suffix('.partial');q.write_text(json.dumps(d,indent=2),encoding='utf-8');q.replace(p)

def main():
    check_generation_config();DATA.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(version=3,seed=20261031,physical_config=CONFIG,bounds_m=BOUNDS,device_clearance_m=CLEARANCE,train_base_per_zone=600,train_jitter_per_zone=120,validation_base_per_zone=90,validation_jitter_per_zone=30,calibration_temperature_per_zone=25,calibration_guard_per_zone=25,test_per_zone=100,boundary_test_per_zone=50,device_test_per_zone=45,material_test_per_zone=30,boundary_width_m=.15,training_device_jitter_m=.1,selection='Model selected only on validation. Calibration and all final tests are disjoint. Old central-only results are historical; no full-coverage claims use those scores.')
    protocolpath=DATA/'protocol.json'
    if protocolpath.exists() and json.loads(protocolpath.read_text(encoding='utf-8'))!=protocol:raise RuntimeError('Protocol changed: use a new experiment directory.')
    protocolpath.write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    rng=np.random.default_rng(protocol['seed']);rows=[];seen=set()
    definitions=[('train_base','train',600,'base'),('train_jitter','train',120,'random'),('val_base','validation',90,'base'),('val_jitter','validation',30,'random'),('cal_temp','cal_temperature',25,'base'),('cal_guard','cal_guard',25,'base'),('test','test',100,'base'),('boundary','boundary_test',50,'base'),('device','device_shift',45,'shift'),('material','material_shift',30,'material')]
    for prefix,split,n,layout in definitions:
        for zone in range(9):
            col=zone%3;row=zone//3
            low=np.array([max(BOUNDS[0],col*2),max(BOUNDS[2],row*2)])
            high=np.array([min(BOUNDS[1],col*2+2),min(BOUNDS[3],row*2+2)])
            for i in range(n):
                targeted=(split=='boundary_test')or(prefix=='train_base' and i%3==0)or(prefix=='val_base' and i%3==0)
                for attempt in range(100000):
                    offsets=rng.uniform(-.1,.1,(8,2)) if layout=='random' else np.zeros((8,2))
                    if layout=='shift':offsets=np.array([[.06,-.04] if j%2==0 else [-.06,.04] for j in list(range(2))+list(range(6))])
                    xy=rng.uniform(low,high)
                    if targeted:
                        options=[]
                        for axis,k in [(0,col),(1,row)]:
                            if k>0:options.append((axis,k*2,1))
                            if k<2:options.append((axis,k*2+2,-1))
                        axis,bound,side=options[rng.integers(len(options))]
                        xy[axis]=bound+side*rng.uniform(.002,.15)
                    key=tuple(xy)
                    if valid_placement(xy,offsets) and key not in seen:break
                else:raise RuntimeError('Cannot sample a physically valid placement')
                seen.add(key)
                rows.append(dict(id=f'v3_{prefix}_Z{zone+1}_{i:04}',split=split,zone=zone,xy=xy.tolist(),device_offset_xy=offsets.tolist(),layout=layout,boundary=bool(near_boundary(xy)),targeted_boundary=targeted,seed=17))
    (DATA/'manifest.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    records=DATA/'records';records.mkdir(exist_ok=True)
    scene,obj,solver=build_scene();obj.position=[3,3,-10]
    np.save(DATA/'empty_reference.npy',response(scene,solver,17))
    active='base';old_offsets=None;start=time.time();done=0
    for r in rows:
        dest=records/(r['id']+'.npz')
        if dest.exists():
            with np.load(dest,allow_pickle=False) as a:
                if a['h'].shape!=(12,64) or not np.isfinite(a['h']).all():raise ValueError('Bad cached channel '+r['id'])
            done+=1;continue
        layout='material' if r['layout']=='material' else 'base'
        if layout!=active:scene,obj,solver=build_scene(material_shift=layout=='material');active=layout;old_offsets=None
        offsets=r['device_offset_xy']
        if offsets!=old_offsets:
            idx=0
            for kind in ['tx','rx']:
                for i,position in enumerate(CONFIG[kind+'_m']):
                    p=np.array(position,float);p[:2]+=offsets[idx];idx+=1;scene.get(f'{kind}{i}').position=p.tolist()
            old_offsets=offsets
        obj.position=[*r['xy'],.6];h=response(scene,solver,17)
        if not np.isfinite(h).all():raise RuntimeError('Nonfinite RT channel')
        np.savez_compressed(dest,h=h);done+=1
        if done%100==0:
            progress(state='generating',done=done,total=len(rows),split=r['split'],elapsed_s=time.time()-start)
            print(done,'/',len(rows),r['split'],round(time.time()-start,1),'s',flush=True)
    progress(state='generated',done=done,total=len(rows),elapsed_s=time.time()-start)
    print('All configurations valid, full-coverage dataset generated:',len(rows),flush=True)

if __name__=='__main__':main()
