
import os,time,json,subprocess,sys,traceback,socket
from pathlib import Path
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from layout_t1_v3_pipeline import O,D,atomic
from layout_t1_v1_common import read,sha
from layout_t1_v3_infer import batch,infer_v5
from layout_t1_v1_infer import batch as old_batch,infer_v5 as old_infer
from layout_t1_v1_evaluate import metrics
from layout_t1_v1_window import window
from home_v5_verify import request
ROOT=Path(__file__).resolve().parent
def main():
 atomic('release_state.json',dict(state='waiting_for_results',pid=os.getpid()))
 while not (O/'results.json').exists():
  if (O/'state.json').exists() and read(O/'state.json')['state']=='failed':raise RuntimeError(read(O/'state.json')['error'])
  time.sleep(5)
 r=read(O/'results.json');assert r['release_check']['eligible'],'Fresh paired improvement criteria failed; retain v1'
 f=read(O/'frozen.json');assert sha(O/'model.joblib')==f['model_sha256']
 for name,digest in f['source_sha256'].items():assert sha(ROOT/name)==digest,name
 assert sha(O/'selection.json')==f['selection_sha256'];assert read(O/'data_audit.json')['passed']
 b=joblib.load(O/'model.joblib');old=joblib.load(ROOT/'outputs/layout_t1_v1/model.joblib');assert np.array_equal(b['reference'],old['reference'])
 with threadpool_limits(limits=3):
  for split in ['calibration','test','boundary_test']:
   with np.load(O/('features_'+split+'.npz')) as z:d={k:z[k] for k in z.files}
   assert metrics(d,batch(b,d))==r['splits'][split],split
   assert metrics(d,old_batch(old,d))==r['baseline_comparison'][split]['previous_t1_model'],split+' baseline'
  rows=read(D/'manifest.json');row=next(a for a in rows if a['split']=='test')
  with np.load(D/'records'/(row['id']+'.npz')) as z:h=window(z['h'],np.random.default_rng(777),30)
  rotated=h*np.exp(1j*np.linspace(0,5*np.pi,len(h)))[:,None,None]
  a=infer_v5(b,rotated);c=old_infer(old,rotated)
  assert a.get('feature_view')=='phase_free' and c.get('feature_view')=='phase_free'
  assert a.get('per_zone_object_count')==c.get('per_zone_object_count'),'Phase fallback must not apply calibrated correction'
  assert infer_v5(b,np.zeros((16,32,64),complex))['status']=='Unknown'
 atomic('evaluation_binding.json',dict(passed=True,model_sha256=f['model_sha256'],results_sha256=sha(O/'results.json'),splits_recomputed=['calibration','test','boundary_test'],phase_fallback_passed=True,zero_signal_unknown=True))
 marker=dict(model_sha256=f['model_sha256'],frozen_sha256=sha(O/'frozen.json'),results_sha256=sha(O/'results.json'),audit_sha256=sha(O/'data_audit.json'),binding_sha256=sha(O/'evaluation_binding.json'))
 atomic('CANDIDATE.json',marker)
 with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
 base='http://127.0.0.1:'+str(port);proc=None
 try:
  with (O/'candidate.stdout.log').open('ab') as stdout,(O/'candidate.stderr.log').open('ab') as stderr:
   proc=subprocess.Popen([sys.executable,'-u',str(ROOT/'app.py'),'--port',str(port)],cwd=ROOT,env=dict(os.environ,ROOMSENSE_VERIFY_T1_V3='1'),stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
  atomic('candidate_server.json',dict(pid=proc.pid,port=port));deadline=time.monotonic()+120
  while True:
   if proc.poll() is not None:raise RuntimeError('Candidate service exited')
   try:
    code,d=request(base,'/api/ready')
    if code==200 and d['ready']:break
   except OSError:pass
   if time.monotonic()>deadline:raise RuntimeError('Candidate startup timeout')
   time.sleep(.5)
  subprocess.run([sys.executable,'layout_t1_v3_verify.py','--base',base],cwd=ROOT,check=True)
  subprocess.run([sys.executable,'home_v5_verify.py','--base',base],cwd=ROOT,check=True)
 finally:
  if proc is not None:
   proc.terminate()
   try:proc.wait(timeout=15)
   except subprocess.TimeoutExpired:proc.kill();proc.wait()
 marker['api_sha256']=sha(O/'api_verification.json');marker['approved_at']=time.time();atomic('APPROVED.json',marker)
 code,d=request('http://127.0.0.1:8769','/api/state');assert code==200 and next(e for e in d['config']['available_layouts'] if e['id']=='t1_v1')['revision']=='t1_count_v3'
 atomic('release_state.json',dict(state='backend_released_ui_pending',model_sha256=f['model_sha256']))
 print('Verified accuracy upgrade activated; browser visual acceptance still pending.',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:atomic('release_state.json',dict(state='failed',error=str(e),traceback=traceback.format_exc()));raise
