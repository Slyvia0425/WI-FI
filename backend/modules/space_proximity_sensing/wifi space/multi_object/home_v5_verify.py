
import argparse,json,time,urllib.request,urllib.error
import numpy as np
from multi_common import SIZES,validate
from home_v5_common import O5,read,atomic,sha,now

def request(base,path,payload=None):
    data=None if payload is None else json.dumps(payload,allow_nan=True).encode();q=urllib.request.Request(base+path,data=data,headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(q,timeout=120) as f:return f.status,json.load(f)
    except urllib.error.HTTPError as e:return e.code,json.load(e)
def main(base):
    checks=[];examples=[]
    def check(name,value):checks.append(dict(name=name,passed=bool(value)));assert value,name
    code,h=request(base,'/api/ready');check('v5 ready',code==200 and h['model_version']==5 and h['ready'])
    code,s=request(base,'/api/state');s['records']=[r for r in s['records'] if r.get('layout_id','default')=='default'];check('v5 catalog3000 scenes',code==200 and s['model_version']==5 and len(s['records'])==3000);check('shared-zone capability',s['config']['supports_shared_zones'] and s['config']['separate_object_count']);check('shared held-out scenes available',sum(r['shared'] and r['split']=='test' for r in s['records'])==1080)
    check('fresh metrics shown',s['occupancy_results']['splits']['test']==read(O5/'results.json')['splits']['test'])
    def consistency(o):
        if o['status']=='Unknown':return o['estimated_object_count'] is None and not o['regions']
        z=o['per_zone_object_count'];return len(z)==9 and sum(z)==o['estimated_object_count'] and sum(c>0 for c in z)==o['estimated_region_count'] and all(r['object_count']==z[r['zone']] for r in o['regions'])
    code,signal=request(base,'/api/example-json');check('signal-only sample16x32x64',code==200 and set(signal)=={'real','imag'} and np.shape(signal['real'])==(16,32,64))
    code,a=request(base,'/api/predict',signal);code,b=request(base,'/api/predict',signal);check('upload has no truth',a['truth'] is None and a['occupancy']['model_version']==5);check('counts internally consistent',consistency(a['occupancy']));check('identical signal deterministic',a['occupancy']['regions']==b['occupancy']['regions'] and a['occupancy']['estimated_object_count']==b['occupancy']['estimated_object_count'])
    code,_=request(base,'/api/predict',dict(signal,objects=[dict(zone=4)],count=3));check('geometry/count hints rejected',code==400)
    zero=dict(real=np.zeros((16,32,64)).tolist(),imag=np.zeros((16,32,64)).tolist());code,z=request(base,'/api/predict',zero);check('zero CSI Unknown',code==200 and z['occupancy']['status']=='Unknown')
    one=dict(real=signal['real'][0],imag=signal['imag'][0]);code,x=request(base,'/api/predict',one);check('single frame not validated as accepted',code==200 and x['occupancy']['status']!='Accepted regions')
    code,_=request(base,'/api/record',dict(id='h5_train_00000'));check('training replay blocked',code==400)
    def obj(k,x,y):return dict(kind=k,x=x,y=y,size=SIZES[k].tolist())
    cases=[('two_in_Z5',[obj(0,3,3.05),obj(2,3,2.15)]),('three_in_Z5',[obj(0,2.45,3.65),obj(1,3.48,3.65),obj(2,3,2.60)])]
    for name,ps in cases:
        validate(ps,allow_shared_zones=True);code,d=request(base,'/api/simulate',dict(objects=ps,seed=42,snr_db=30));check(name+' accepted by physical simulator',code==200 and d['true_count']==len(ps) and d['true_region_count']==1);check(name+' prediction consistency',consistency(d['occupancy']));examples.append(dict(name=name,truth_objects=len(ps),truth_zones=1,prediction=d['occupancy']))
    code,one=request(base,'/api/simulate',dict(objects=[obj(0,2.65,2.75)],seed=42,snr_db=30));check('single target with fixed cabinet',code==200 and one['true_count']==1 and consistency(one['occupancy']))
    bad=[obj(0,3,5.72)];code,_=request(base,'/api/simulate',dict(objects=bad));check('fixed background collision rejected',code==400)
    missing=dict(real=signal['real'],imag=signal['imag']);missing['real']=np.array(missing['real']);missing['imag']=np.array(missing['imag']);missing['real'][:,:16]=0;missing['imag'][:,:16]=0;missing={k:v.tolist() for k,v in missing.items()};code,d=request(base,'/api/predict',missing);check('missing required links Unknown',code==200 and d['occupancy']['status']=='Unknown' and d['occupancy']['reason']=='missing_required_links')
    bad=[obj(0,3,3),obj(2,5.15,3.4)];code,_=request(base,'/api/simulate',dict(objects=bad));check('direct API still rejects radio obstruction',code==400)
    bad=[obj(0,3,3),obj(2,3,3)];code,_=request(base,'/api/simulate',dict(objects=bad));check('physical interpenetration still rejected',code==400)
    for count in [2,3]:
        row=next(r for r in s['records'] if r['split']=='test' and r['shared'] and r['count']==count);code,d=request(base,'/api/record',dict(id=row['id'],seed=52,snr_db=30));check('shared '+str(count)+' replay works',code==200 and consistency(d['occupancy']))
    check('frozen weights unchanged',sha(O5/'model.joblib')==read(O5/'frozen.json')['model_sha256'])
    atomic('verification.json',dict(passed=True,completed_at=now(),base=base,model_sha256=sha(O5/'model.joblib'),checks=checks,example_diagnostics=examples,note='Examples are functional checks, not independent accuracy estimates. Prediction mistakes are preserved.'));print(json.dumps(dict(passed=True,checks=len(checks),examples=[dict(name=e['name'],pred=e['prediction'].get('per_zone_object_count'),status=e['prediction']['status']) for e in examples]),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',default='http://127.0.0.1:8773');a=p.parse_args();main(a.base)
