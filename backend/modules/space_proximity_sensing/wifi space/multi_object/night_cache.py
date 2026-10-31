
import os,json,time,hashlib,argparse
from pathlib import Path
from datetime import datetime
import numpy as np
from multi_common import DATA,target
from opt_v2_common import DATA2,targets,window
from night_data import D3
from night_features import views,fingerprint,SPEC
from night_common import NIGHT,atomic,stage_lock,log,now,DEADLINE,TZ,alive
from multi_train import seed
CACHE=NIGHT/'cache';CACHE.mkdir(exist_ok=True)
QUALITY=['phase_health','temporal_coherence','roughness_cal','roughness_free','calibrated_usable']

def jobs_for(split):
    jobs=[]
    if split=='train':
        for folder,prefix in [(DATA,'old'),(DATA2,'v2')]:
            for r in json.loads((folder/'manifest.json').read_text()):
                if r['split']=='train':jobs.append((folder,dict(r,group_id=prefix+':'+r['id'])))
    jobs.extend((D3,r) for r in json.loads((D3/'manifest.json').read_text()) if r['split']==split)
    return jobs

def expected_meta(split,reference):
    jobs=jobs_for(split);repeat=2 if split=='train' else 1
    descriptor=[dict(folder=p.name,id=r['id'],group_id=r['group_id'],objects=r['objects']) for p,r in jobs]
    meta=dict(cache_version=1,split=split,physical_scenes=len(jobs),observations=len(jobs)*repeat,repeats=repeat,feature_spec=SPEC,feature_fingerprint=fingerprint(reference),manifest_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),seed_policy='night-view-v1:<background-group>:<repeat>; paired train records share receiver randomness, never group IDs as features.',quality_fields=QUALITY)
    meta['fingerprint']=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest();return meta,jobs

def write_json(path,value):
    tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');tmp.replace(path)

def build(split):
    if split in ['test','boundary_test'] and not (NIGHT/'candidate_frozen.json').exists():raise RuntimeError('Final test is sealed until candidate_frozen.json exists')
    with stage_lock('cache_'+split):
        ref=np.load(DATA/'empty_reference.npy');meta,jobs=expected_meta(split,ref);folder=CACHE/split;folder.mkdir(exist_ok=True)
        mp=folder/'meta.json'
        if mp.exists() and json.loads(mp.read_text())!=meta:raise RuntimeError('Cache fingerprint changed: use a new version/directory, do not mix features')
        if (folder/'complete.json').exists():
            done=json.loads((folder/'complete.json').read_text());assert done['fingerprint']==meta['fingerprint'];return folder
        write_json(mp,meta);n=meta['observations'];repeat=meta['repeats']
        arrays={}
        for name,dim in [('calibrated',1600),('phase_free',1088),('quality',len(QUALITY))]:
            p=folder/(name+'.npy');arrays[name]=np.lib.format.open_memmap(p,mode='r+' if p.exists() else 'w+',dtype=np.float32,shape=(n,dim))
            if arrays[name].shape!=(n,dim) or arrays[name].dtype!=np.float32:raise RuntimeError('Unexpected cache shape')
        rows=[r for _,r in jobs]
        targets_by_name=dict(labels=targets(rows),typed=np.array([target(r['objects']) for r in rows]),ids=np.array([r['id'] for r in rows]),groups=np.array([r['group_id'] for r in rows]))
        for name,array in targets_by_name.items():
            path=folder/(name+'.npy')
            if not path.exists():
                tmp=path.with_suffix('.partial')
                with tmp.open('wb') as f:np.save(f,np.repeat(array,repeat,axis=0))
                tmp.replace(path)
            if np.load(path,mmap_mode='r',allow_pickle=False).shape[0]!=n:raise RuntimeError('Incomplete target cache: '+name)
        progress=folder/'progress.json';next_i=json.loads(progress.read_text())['next_geometry'] if progress.exists() else 0;started=time.time();last_wait=None
        for index in range(next_i,len(jobs)):
            if datetime.now(TZ)>=DEADLINE:raise RuntimeError('Night acquisition deadline reached; partial cache preserved')
            source,row=jobs[index];path=source/'records'/(row['id']+'.npz')
            while not path.exists():
                if last_wait!=row['id']:atomic('cache_'+split+'_state.json',dict(state='waiting_for_data',pid=os.getpid(),updated=now(),next_geometry=index,total=len(jobs),id=row['id']));last_wait=row['id']
                state_path=NIGHT/'data_state.json'
                if state_path.exists():
                    state=json.loads(state_path.read_text())
                    if state['state'] in ['failed','deadline_stopped']:raise RuntimeError('Data generator stopped: '+str(state))
                if datetime.now(TZ)>=DEADLINE:raise RuntimeError('Deadline while waiting for data')
                time.sleep(3)
            with np.load(path,allow_pickle=False) as a:h=a['h']
            if h.shape!=(32,64) or not np.isfinite(h).all():raise ValueError('Invalid raw record '+row['id'])
            for k in range(repeat):
                rng=np.random.default_rng(seed('night-view-v1:'+row['group_id']+':'+str(k)));obs=window(h,rng);v=views(obs,ref);j=index*repeat+k
                arrays['calibrated'][j]=v['calibrated'];arrays['phase_free'][j]=v['phase_free'];arrays['quality'][j]=[v[q] for q in QUALITY]
            if (index+1)%128==0 or index+1==len(jobs):
                for arr in arrays.values():arr.flush()
                write_json(progress,dict(next_geometry=index+1,fingerprint=meta['fingerprint']));atomic('cache_'+split+'_state.json',dict(state='extracting',pid=os.getpid(),updated=now(),done=index+1,total=len(jobs),elapsed_s=time.time()-started))
        hashes={}
        for name in ['calibrated','phase_free','quality','labels','typed','ids','groups']:
            p=folder/(name+'.npy');hh=hashlib.sha256()
            with p.open('rb') as f:
                for chunk in iter(lambda:f.read(8*1024*1024),b''):hh.update(chunk)
            hashes[name]=hh.hexdigest()
        write_json(folder/'complete.json',dict(fingerprint=meta['fingerprint'],finished=now(),array_sha256=hashes));atomic('cache_'+split+'_state.json',dict(state='completed',pid=os.getpid(),updated=now(),physical_scenes=len(jobs),observations=n,elapsed_s=time.time()-started));log('Cache complete: '+split+', '+str(len(jobs))+' physical scenes, '+str(n)+' observations, calibrated1600/free1088.');return folder

def load(split,view=None,wait=False):
    folder=CACHE/split
    while not (folder/'complete.json').exists():
        if not wait:raise RuntimeError('Cache not complete: '+split)
        if datetime.now(TZ)>=DEADLINE:raise RuntimeError('Deadline while waiting for cache')
        state_path=NIGHT/('cache_'+split+'_state.json')
        if state_path.exists():
            state=json.loads(state_path.read_text())
            if state['state']=='failed':raise RuntimeError('Cache failed: '+str(state))
        time.sleep(3)
    ref=np.load(DATA/'empty_reference.npy');expected,_=expected_meta(split,ref);meta=json.loads((folder/'meta.json').read_text());complete=json.loads((folder/'complete.json').read_text())
    if meta!=expected or complete['fingerprint']!=expected['fingerprint']:raise RuntimeError('Cache provenance mismatch')
    names=['labels','typed','ids','groups','quality']+([view] if view else ['calibrated','phase_free'])
    return dict(meta=meta,**{name:np.load(folder/(name+'.npy'),mmap_mode='r',allow_pickle=False) for name in names})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('splits',nargs='*',default=['train','validation','calibration']);args=ap.parse_args()
    for split in args.splits:
        try:build(split)
        except Exception as e:
            import traceback
            atomic('cache_'+split+'_state.json',dict(state='failed',pid=os.getpid(),updated=now(),error=str(e),traceback=traceback.format_exc()));raise
