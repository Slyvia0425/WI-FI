
import os,time,json,hashlib,argparse
from pathlib import Path
import numpy as np
from multi_common import DATA
from layout_t1_v1_common import D5,O5,atomic,read,lock,sha,now,labels,zone_counts
from night_features import views,fingerprint
from night_cache import jobs_for as old_jobs,load as old_load,QUALITY
from layout_t1_v1_window import window
from multi_train import seed
CACHE=O5/'cache'

def jobs_for(split):
    while not (D5/'manifest.json').exists():
        status=O5/'data_state.json'
        if status.exists() and read(status)['state']=='failed':raise RuntimeError('Data generation failed')
        time.sleep(3)
    jobs=[]
    jobs.extend((D5,r) for r in read(D5/'manifest.json') if r['split']==split)
    return jobs

def build(split):
    if split in ['test','boundary_test'] and not (O5/'frozen.json').exists():raise RuntimeError('Final tests sealed until model frozen')
    with lock('cache_'+split):
        jobs=jobs_for(split)
        while not (D5/'empty_reference.npy').exists():time.sleep(3)
        ref=np.load(D5/'empty_reference.npy');repeat=2 if split=='train' else 1;n=len(jobs)*repeat;folder=CACHE/split;folder.mkdir(exist_ok=True,parents=True)
        meta=dict(version=5,split=split,physical_scenes=len(jobs),observations=n,repeats=repeat,features_sha256=fingerprint(ref),receiver_sha256=sha(Path(__file__).with_name('layout_t1_v1_window.py')),physics_sha256=sha(Path(__file__).with_name('layout_t1_v1_physics.py')),manifest_sha256=hashlib.sha256(json.dumps([dict(id=r['id'],group_id=r['group_id'],objects=r['objects']) for _,r in jobs],sort_keys=True).encode()).hexdigest(),seed_policy='layout-t1-v1-view:<group_id>:<repeat>; no reused old physical scenes.',quality_fields=QUALITY)
        meta['fingerprint']=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
        if (folder/'complete.json').exists():assert read(folder/'meta.json')==meta;return folder
        if (folder/'meta.json').exists():assert read(folder/'meta.json')==meta
        (folder/'meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');arrays={}
        for key,dim in [('calibrated',1600),('phase_free',1088),('quality',5)]:
            p=folder/(key+'.npy');arrays[key]=np.lib.format.open_memmap(p,mode='r+' if p.exists() else 'w+',dtype=np.float32,shape=(n,dim));assert arrays[key].shape==(n,dim)
        rows=[r for _,r in jobs];y,counts=labels(rows)
        targets=dict(labels=y,zone_count=zone_counts(rows),object_count=counts,ids=np.array([r['id'] for r in rows]),groups=np.array([r['group_id'] for r in rows]),shared=counts>y.sum(1))
        for key,a in targets.items():
            p=folder/(key+'.npy')
            if not p.exists():
                with p.with_suffix('.partial').open('wb') as f:np.save(f,np.repeat(a,repeat,axis=0))
                p.with_suffix('.partial').replace(p)
        progress=folder/'progress.json';done=read(progress)['next_geometry'] if progress.exists() else 0;start=time.time();old_n=sum(p!=D5 for p,r in jobs)
        for i in range(done,len(jobs)):
            folder_src,r=jobs[i];p=folder_src/'records'/(r['id']+'.npz')
            while not p.exists():
                status=O5/'data_state.json'
                if status.exists() and read(status)['state']=='failed':raise RuntimeError('Data generator failed')
                atomic('cache_'+split+'_state.json',dict(state='waiting_for_signal',pid=os.getpid(),updated=now(),done=i,total=len(jobs)));time.sleep(3)
            with np.load(p,allow_pickle=False) as a:h=a['h']
            assert h.shape==(32,64) and np.isfinite(h).all()
            for k in range(repeat):
                prefix='layout-t1-v1-view:'
                obs=window(h,np.random.default_rng(seed(prefix+r['group_id']+':'+str(k))));v=views(obs,ref);j=i*repeat+k
                for key in ['calibrated','phase_free']:arrays[key][j]=v[key]
                arrays['quality'][j]=[v[q] for q in QUALITY]
            if (i+1)%128==0 or i+1==len(jobs):
                for arr in arrays.values():arr.flush()
                tmp=progress.with_suffix('.partial');tmp.write_text(json.dumps(dict(next_geometry=i+1)),encoding='utf-8');tmp.replace(progress);atomic('cache_'+split+'_state.json',dict(state='extracting',pid=os.getpid(),updated=now(),done=i+1,total=len(jobs),elapsed_s=time.time()-start))
        complete=dict(fingerprint=meta['fingerprint'],completed_at=now(),array_sha256={p.stem:sha(p) for p in folder.glob('*.npy')});(folder/'complete.json').write_text(json.dumps(complete,indent=2),encoding='utf-8');atomic('cache_'+split+'_state.json',dict(state='completed',pid=os.getpid(),updated=now(),physical_scenes=len(jobs),observations=n));return folder

def load(split,wait=False):
    folder=CACHE/split
    while not (folder/'complete.json').exists():
        if not wait:raise RuntimeError('Cache not ready: '+split)
        status=O5/('cache_'+split+'_state.json')
        if status.exists() and read(status)['state']=='failed':raise RuntimeError('Cache failed: '+split)
        time.sleep(3)
    meta=read(folder/'meta.json');assert meta['features_sha256']==fingerprint(np.load(D5/'empty_reference.npy'))
    assert meta['receiver_sha256']==sha(Path(__file__).with_name('layout_t1_v1_window.py'))
    assert meta['physics_sha256']==sha(Path(__file__).with_name('layout_t1_v1_physics.py'))
    return dict(meta=meta,**{p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in folder.glob('*.npy')})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('splits',nargs='*',default=['train','validation','calibration']);a=p.parse_args()
    for split in a.splits:
        try:build(split)
        except Exception as e:
            import traceback
            atomic('cache_'+split+'_state.json',dict(state='failed',pid=os.getpid(),updated=now(),error=str(e),traceback=traceback.format_exc()));raise
