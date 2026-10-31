
import json,os,time,tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from opt_v2_common import DATA2,OUT2

def worker(rows):

    folder=Path(tempfile.gettempdir())/('wifi_rt_worker_'+str(os.getpid()));folder.mkdir(exist_ok=True);tempfile.tempdir=str(folder)
    from multi_simulate import build,place,channel
    import gc
    sim=None;used=0;done=0
    for row in rows:
        path=DATA2/'records'/(row['id']+'.npz')
        if path.exists():continue
        if sim is None or used>=100:
            sim=None;gc.collect();sim=build();used=0
        scene,objects,solver=sim;place(objects,row['objects']);h=channel(scene,solver);used+=1
        tmp=path.with_suffix('.'+str(os.getpid())+'.partial')
        with tmp.open('wb') as f:np.savez_compressed(f,h=h)
        tmp.replace(path);done+=1
    return done

def main():
    rows=json.loads((DATA2/'manifest.json').read_text());missing=[r for r in rows if not (DATA2/'records'/(r['id']+'.npz')).exists()];start=time.time();before=len(rows)-len(missing)
    print('Resume',before,'finished;',len(missing),'remaining; 3 workers',flush=True)
    with ProcessPoolExecutor(max_workers=3) as pool:
        fs=[pool.submit(worker,missing[k::3]) for k in range(3)]
        while not all(f.done() for f in fs):
            for f in fs:
                if f.done():f.result()
            count=len(list((DATA2/'records').glob('*.npz')));p=OUT2/'progress.json';tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(dict(state='generating_parallel',done=count,total=len(rows),workers=3,elapsed_s=time.time()-start),indent=2));tmp.replace(p);time.sleep(5)
        print('Worker totals',[f.result() for f in fs],flush=True)
    (OUT2/'progress.json').write_text(json.dumps(dict(state='generated',done=len(rows),total=len(rows),elapsed_s=time.time()-start),indent=2))
if __name__=='__main__':main()
