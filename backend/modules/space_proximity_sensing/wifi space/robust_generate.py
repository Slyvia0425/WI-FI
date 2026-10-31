
import json,time,shutil
import numpy as np
from simulate import ROOT,CONFIG,build_scene,response,status,check_generation_config

def main():
    check_generation_config()
    data=ROOT/'data';archive=ROOT/'outputs/phase1';archive.mkdir(parents=True,exist_ok=True)
    phase1_manifest=data/'manifest_phase1.json'
    if not phase1_manifest.exists():
        shutil.copyfile(data/'manifest.json',phase1_manifest)
        for name in ['model.joblib','results.json','predictions.json']:
            if (ROOT/'outputs'/name).exists():shutil.copyfile(ROOT/'outputs'/name,archive/name)
    base=json.loads(phase1_manifest.read_text())
    for row in base:
        if row['split'] in ['test','device_shift','material_shift']:row['split']='diagnostic_original_'+row['split']
    protocol=dict(version=2,seed=20261030,train_jitter_positions_per_zone=180,validation_jitter_positions_per_zone=30,test_positions_per_zone=45,shifted_test_positions_per_zone=30,material_test_positions_per_zone=15,device_jitter_m=.1,selection='All model/hyperparameter choices use only joint validation. Phase-1 tests are diagnostic; all final tests use new positions and are not selected on.',reference='Original fixed-device empty room; no per-object or test-label calibration')

    (data/'protocol_v2.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    rng=np.random.default_rng(protocol['seed']);jobs=[]
    definitions=[('train_jitter','train',180,'random'),('val_jitter','validation',30,'random'),('final_test','test',45,'base'),('final_device','device_shift',30,'shift'),('final_material','material_shift',15,'material')]
    for prefix,split,n,layout in definitions:
        for zone in range(9):
            for i in range(n):
                xy=np.array([1+2*(zone%3),1+2*(zone//3)])+rng.uniform(-np.array(CONFIG['position_jitter_m']),np.array(CONFIG['position_jitter_m']))
                offsets=rng.uniform(-.1,.1,(8,2)) if layout=='random' else np.zeros((8,2))
                if layout=='shift':offsets=np.array([[.06,-.04] if i%2==0 else [-.06,.04] for i in list(range(2))+list(range(6))])
                jobs.append(dict(id=f'{prefix}_Z{zone+1}_{i:03}',split=split,zone=zone,xy=xy.tolist(),device_offset_xy=offsets.tolist(),layout=layout,seed=17))
    allrows=base+jobs
    (data/'manifest.json').write_text(json.dumps(allrows,indent=2),encoding='utf-8')
    scene,obj,solver=build_scene();active='base';start=time.time();done=0
    for row in jobs:
        p=data/'records'/(row['id']+'.npz')
        if p.exists():
            with np.load(p,allow_pickle=False) as a:
                if a['h'].shape!=(12,64) or not np.isfinite(a['h']).all():raise ValueError('Bad cached channel '+row['id'])
            done+=1;continue
        layout='material' if row['layout']=='material' else 'base'
        if active!=layout:scene,obj,solver=build_scene(material_shift=layout=='material');active=layout
        offsets=row['device_offset_xy'];idx=0
        for kind,positions in [('tx',CONFIG['tx_m']),('rx',CONFIG['rx_m'])]:
            for i,position in enumerate(positions):
                q=np.array(position,dtype=float);q[:2]+=offsets[idx];idx+=1;scene.get(f'{kind}{i}').position=q.tolist()
        obj.position=[*row['xy'],.6];h=response(scene,solver,17)
        if not np.isfinite(h).all():raise ValueError('Bad RT signal '+row['id'])
        np.savez_compressed(p,h=h);done+=1
        status(state='domain_randomization',done=done,total=len(jobs),split=row['split'],elapsed_s=time.time()-start)
        if done%100==0:print(done,'/',len(jobs),'phase2',row['split'],round(time.time()-start,1),'s',flush=True)
    status(state='generated',done=len(allrows),total=len(allrows),elapsed_s=time.time()-start,engine='Sionna RT 1.2.2',protocol=2)
    print('Total physical configurations:',len(allrows),flush=True)

if __name__=='__main__':main()
