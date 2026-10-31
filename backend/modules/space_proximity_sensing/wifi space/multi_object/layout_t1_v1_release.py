
import argparse,json,os,subprocess,sys,time,traceback
from urllib.parse import urlparse
from collections import defaultdict
from pathlib import Path
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from layout_t1_v1_common import D5,O5,read,atomic,sha,now,lock

ROOT=Path(__file__).resolve().parent

def wait_for(path,stages):
    while not path.exists():
        for stage in stages:
            p=O5/(stage+'_state.json')
            if p.exists() and read(p).get('state')=='failed':
                raise RuntimeError(stage+': '+read(p).get('error','Failed'))
        time.sleep(5)

def offline():
    frozen=read(O5/'frozen.json');results=read(O5/'results.json')
    assert results['release_check']['eligible'],'Final test release criteria not met'
    assert sha(O5/'model.joblib')==frozen['model_sha256']
    for name,digest in frozen['source_sha256'].items():assert sha(ROOT/name)==digest,name
    subprocess.run([sys.executable,str(ROOT/'layout_t1_v1_audit.py'),'--files'],cwd=ROOT,check=True)
    audit=read(O5/'data_audit_full.json');assert audit['clean_split_audit']
    rows=read(D5/'manifest.json');geom=defaultdict(set)
    for r in rows:
        key=json.dumps(sorted((o['kind'],o['x'],o['y'],*o['size']) for o in r['objects']))
        geom[key].add(r['split'])
    assert not any(len(v)>1 for v in geom.values()),'Exact geometry shared across splits'
    b=joblib.load(O5/'model.joblib');assert np.array_equal(b['reference'],np.load(D5/'empty_reference.npy'))


    from layout_t1_v1_evaluate import metrics
    from layout_t1_v1_cache import load
    from layout_t1_v1_infer import batch
    recomputed={}
    with threadpool_limits(limits=4):
        for split in ['validation','calibration','test','boundary_test']:
            d=load(split);mm=metrics(d,batch(b,d));assert mm==results['splits'][split],split+' stale metrics'
            recomputed[split]=mm
    m=recomputed['test']
    assert m['occupied_zone_macro_f1']>=.85 and m['count_accuracy']>=.85
    assert m['exact_joint_accuracy']>=.65 and m['by_sharing']['shared']['count_accuracy']>=.80
    atomic('evaluation_binding.json',dict(passed=True,checked_at=now(),model_sha256=frozen['model_sha256'],results_sha256=sha(O5/'results.json'),frozen_sha256=sha(O5/'frozen.json'),splits=list(recomputed),scope='All reported metrics recomputed with current frozen weights; no fitting.'))
    from layout_t1_v1_window import window
    from layout_t1_v1_infer import infer_v5
    from home_v5_infer import infer_v5 as original_schema
    checks=[]
    chosen=[]
    for n in [1,2,3]:
        for shared in [False,True]:
            chosen.extend([r for r in rows if r['split']=='test' and r['count']==n and r['shared']==shared][:3])
    with threadpool_limits(limits=2):
        for i,r in enumerate(chosen):
            with np.load(D5/'records'/(r['id']+'.npz')) as a:h=window(a['h'],np.random.default_rng(1700+i),30)
            x=infer_v5(b,h);y=original_schema(b,h);z=infer_v5(b,h)
            for k in ['regions','probabilities','status','estimated_object_count','confirmed_objects']:
                assert x[k]==y[k]==z[k],(r['id'],k)
            if x['status']!='Unknown':
                counts=x['per_zone_object_count'];assert sum(counts)==x['estimated_object_count']
                assert sum(c>0 for c in counts)==x['estimated_region_count']
            checks.append(dict(id=r['id'],status=x['status'],predicted_count=x['estimated_object_count']))
    report=dict(passed=True,completed_at=now(),model_sha256=frozen['model_sha256'],source_hashes_passed=True,raw_audit_passed=True,geometry_split_overlap=0,reference_matches_training=True,schema_and_repeatability_checks=checks,scope='Functional checks, not new accuracy estimates. No fitting or threshold adjustment.')
    atomic('offline_verification.json',report)
    return frozen

def main(base):
    with lock('release'):
        atomic('release_state.json',dict(state='waiting_for_evaluation',pid=os.getpid(),updated=now()))
        wait_for(O5/'results.json',['data','training','evaluation'])
        while read(O5/'evaluation_state.json')['state']!='completed':
            state=read(O5/'evaluation_state.json');assert state['state']!='failed',state
            time.sleep(3)
        atomic('release_state.json',dict(state='auditing',pid=os.getpid(),updated=now()))
        frozen=offline()
        approval=dict(layout_id='t1_v1',model_sha256=frozen['model_sha256'],approved_at=now(),frozen_sha256=sha(O5/'frozen.json'),audit_sha256=sha(O5/'data_audit_full.json'),results_sha256=sha(O5/'results.json'),offline_sha256=sha(O5/'offline_verification.json'),evaluation_binding_sha256=sha(O5/'evaluation_binding.json'),scope='Single fixed changed-T1 simulation layout only. Arbitrary transmitter positions remain unvalidated.')

        atomic('CANDIDATE.json',approval)
        import socket
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        candidate_base='http://127.0.0.1:'+str(port)
        env=dict(os.environ,ROOMSENSE_VERIFY_T1='1')
        proc=None
        try:
            from home_v5_verify import request
            with (O5/'candidate.stdout.log').open('ab') as out,(O5/'candidate.stderr.log').open('ab') as err:
                proc=subprocess.Popen([sys.executable,'-u',str(ROOT/'app.py'),'--port',str(port)],cwd=ROOT,env=env,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            atomic('candidate_server.json',dict(pid=proc.pid,port=port,started=now()))
            deadline=time.monotonic()+120
            while True:
                if proc.poll() is not None:raise RuntimeError('Candidate service exited')
                try:
                    code,health=request(candidate_base,'/api/ready')
                    if code==200 and health['ready']:break
                except (OSError,ValueError):pass
                if time.monotonic()>deadline:raise RuntimeError('Candidate service startup timeout')
                time.sleep(.5)
            subprocess.run([sys.executable,str(ROOT/'layout_t1_v1_verify.py'),'--base',candidate_base],cwd=ROOT,check=True)
            subprocess.run([sys.executable,str(ROOT/'home_v5_verify.py'),'--base',candidate_base],cwd=ROOT,check=True)
        finally:
            if proc is not None:
                proc.terminate()
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
        approval['api_sha256']=sha(O5/'api_verification.json');approval['approved_at']=now()
        atomic('APPROVED.json',approval)
        from home_v5_verify import request
        code,public=request(base,'/api/state')
        assert code==200 and next(e for e in public['config']['available_layouts'] if e['id']=='t1_v1')['ready'],'Public service did not discover approved preset'
        atomic('release_state.json',dict(state='api_verified_ui_pending',pid=os.getpid(),updated=now(),model_sha256=frozen['model_sha256']))
        print('Offline/API release passed. Browser acceptance still required.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',default='http://127.0.0.1:8769');a=p.parse_args()
    try:main(a.base)
    except Exception as e:
        atomic('release_state.json',dict(state='failed',pid=os.getpid(),updated=now(),error=str(e),traceback=traceback.format_exc()));raise
