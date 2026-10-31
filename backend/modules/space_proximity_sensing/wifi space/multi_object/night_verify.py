
import argparse,json,hashlib,time,urllib.request,urllib.error
import numpy as np
from night_common import NIGHT,atomic,now
from night_features import views
from night_infer import infer_v3
import joblib

def request(base,path,payload=None):
    data=None if payload is None else json.dumps(payload,allow_nan=True).encode()
    req=urllib.request.Request(base+path,data=data,headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=90) as f:return f.status,json.load(f)
    except urllib.error.HTTPError as e:return e.code,json.load(e)

def main(base,expected):
    checks=[]
    def check(name,condition,detail=None):
        checks.append(dict(name=name,passed=bool(condition),detail=detail));assert condition,name
    code,health=request(base,'/api/ready');check('matching warm service',code==200 and health.get('service')=='roomsense-multi' and health.get('model_version')==expected)
    _,state=request(base,'/api/state');check('ready UI catalog and version',state['ready'] and state['model_version']==expected and len(state['records'])==1950)
    prefix='v3_' if expected==3 else 'v2_';check('only active held-out scenes in catalog',all(r['id'].startswith(prefix) and r['split'] in ['test','boundary_test'] for r in state['records']))
    root=NIGHT if expected==3 else NIGHT.parent/'optimized_v2';path=root/'model.joblib';hh=hashlib.sha256(path.read_bytes()).hexdigest()
    check('UI scores match current model report',state['occupancy_results']['splits']['test']==json.loads((root/'results.json').read_text(encoding='utf-8'))['splits']['test'])
    _,signal=request(base,'/api/example-json');h=np.asarray(signal['real'])+1j*np.asarray(signal['imag']);check('example is signal-only 16-frame CSI',set(signal)=={'real','imag'} and h.shape==(16,32,64) and np.isfinite(h).all())
    code,d=request(base,'/api/predict',signal);check('signal upload has no truth',code==200 and d['truth'] is None and d['occupancy']['model_version']==expected)
    code,d2=request(base,'/api/predict',signal);check('same CSI gives identical predictions',d['occupancy']['regions']==d2['occupancy']['regions'] and d['occupancy']['layout_confidence']==d2['occupancy']['layout_confidence'])
    zero=dict(real=np.zeros((16,32,64)).tolist(),imag=np.zeros((16,32,64)).tolist());code,z=request(base,'/api/predict',zero);check('zero signal remains Unknown',code==200 and z['occupancy']['status']=='Unknown')
    one=dict(real=h.real[0].tolist(),imag=h.imag[0].tolist());code,o=request(base,'/api/predict',one);check('single frame never marked 16-frame accepted',code==200 and o['frames']==1 and o['occupancy']['status']!='Accepted regions')
    code,_=request(base,'/api/predict',dict(signal,objects=[dict(x=1,y=1)],count=2));check('metadata cannot enter signal inference',code==400)
    for name,payload in [('bad shape',dict(real=[[1]],imag=[[0]])),('NaN',dict(real=np.full((32,64),np.nan).tolist(),imag=np.zeros((32,64)).tolist()))]:
        code,_=request(base,'/api/predict',payload);check(name+' rejected',code==400)
    code,_=request(base,'/api/record',dict(id='v3_train_uniform_00000'));check('training record blocked from demo replay',code==400)
    for count in [2,3]:
        row=next(r for r in state['records'] if r['count']==count and r['split']=='test');code,out=request(base,'/api/record',dict(id=row['id'],seed=874,snr_db=30));check(str(count)+'-object held-out replay',code==200 and out['true_count']==count and out['frames']==16 and out['occupancy']['estimated_region_count'] in [1,2,3])
    row=state['records'][0];code,low=request(base,'/api/record',dict(id=row['id'],seed=874,snr_db=0));check('very low quality input not confidently accepted',code==200 and low['occupancy']['status']!='Accepted regions')
    b=joblib.load(path)
    if expected==3:
        check('deployed weights match frozen candidate',hh==json.loads((NIGHT/'candidate_frozen.json').read_text(encoding='utf-8'))['model_sha256'])
        v=views(h,b['reference']);rng=np.random.default_rng(2026103010);rot=h*np.exp(1j*rng.uniform(-np.pi,np.pi,(16,32,1)));w=views(rot,b['reference']);check('phase-free features invariant',np.max(np.abs(v['phase_free']-w['phase_free']))<1e-4)
        rd=infer_v3(b,rot);check('phase failure routes to phase-free expert',rd.get('feature_view')=='phase_free' and rd['estimated_region_count'] in [1,2,3])
        check('raw data audit has no train/evaluation overlaps',all(v==0 for v in json.loads((NIGHT/'data_audit_full.json').read_text(encoding='utf-8'))['train_eval_raw_channel_overlap'].values()))
    result=dict(passed=all(c['passed'] for c in checks),completed_at=now(),base=base,version=expected,model_sha256=hh,checks=checks)
    atomic('verification.json' if expected==3 else 'v2_launcher_verification.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',default='http://127.0.0.1:8770');p.add_argument('--version',type=int,default=3);a=p.parse_args();main(a.base,a.version)
