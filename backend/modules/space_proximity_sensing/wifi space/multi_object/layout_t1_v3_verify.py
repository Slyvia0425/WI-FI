
import argparse,json
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from home_v5_verify import request
from layout_t1_v1_common import read,sha,now
from layout_t1_v3_pipeline import O as O5,atomic
from layout_t1_v3_infer import infer_v5
from layout_registry import NEW_TX
from multi_common import TX,SIZES

def main(base,output="api_verification.json"):
    checks=[]
    def check(name,ok):
        checks.append(dict(name=name,passed=bool(ok)));assert ok,name
    code,s=request(base,'/api/state');check('API ready',code==200 and s['ready'])
    layouts={e['id']:e for e in s['config']['available_layouts']}
    check('Both fixed layouts ready',all(layouts[k]['ready'] for k in ['default','t1_v1']))
    for lid in ['default','t1_v1']:
        rr=[r for r in s['records'] if r['layout_id']==lid]
        check(lid+' has 2400 test plus600 boundary',sum(r['split']=='test' for r in rr)==2400 and sum(r['split']=='boundary_test' for r in rr)==600)
    check('New-layout scores bound to new evaluation',layouts['t1_v1']['evaluation']['splits']['test']==read(O5/'results.json')['splits']['test'])
    b=joblib.load(O5/'model.joblib')
    with threadpool_limits(limits=2):
        for lid,tx in [('default',TX),('t1_v1',NEW_TX)]:
            code,sig=request(base,'/api/example-json?layout_id='+lid)
            check(lid+' correct signal sample metadata',code==200 and np.shape(sig['real'])==(16,32,64) and (sig.get('layout_id','default')==lid))
            code,a=request(base,'/api/predict',sig);_,again=request(base,'/api/predict',sig)
            check(lid+' upload uses matched model and has no geometry truth',code==200 and a['truth'] is None and a['model_layout_id']==lid and a['tx_positions']==tx)
            check(lid+' deterministic upload',a['occupancy']['probabilities']==again['occupancy']['probabilities'])
            if lid=='t1_v1':
                local=infer_v5(b,np.asarray(sig['real'])+1j*np.asarray(sig['imag']))
                check('Server matches independently loaded frozen inference',np.allclose(local['probabilities'],a['occupancy']['probabilities'],rtol=0,atol=1e-9) and local['status']==a['occupancy']['status'])
            code,_=request(base,'/api/predict',dict(sig,objects=[{'zone':4}]))
            check(lid+' upload rejects target hints',code==400)
            row=next(r for r in s['records'] if r['layout_id']==lid and r['split']=='test' and r['shared'] and r['count']==3)
            code,replay=request(base,'/api/record',dict(id=row['id'],layout_id=lid,seed=112,snr_db=30))
            check(lid+' shared three replay uses correct model',code==200 and replay['model_layout_id']==lid and replay['tx_positions']==tx and replay['true_count']==3)
            code,_=request(base,'/api/record',dict(id=row['id'],layout_id='t1_v1' if lid=='default' else 'default'))
            check(lid+' mismatched replay model rejected',code==400)
    for path,payload in [('/api/record',dict(id='t1_train_00000')),('/api/predict',dict(sig,layout_id='other'))]:
        code,_=request(base,path,payload);check(path+' rejects unavailable model or training record',code==400)
    code,_=request(base,'/api/example-json?layout_id=other');check('Unknown example layout rejected',code==400)
    objects=[dict(kind=0,x=2.65,y=2.75,size=SIZES[0].tolist())]
    simulations=[]
    arbitrary=[p[:] for p in NEW_TX];arbitrary[0]=[1.45,1.2,.45]
    for lid,tx in [('t1_v1',NEW_TX),('default',TX),('unvalidated',arbitrary),('t1_v1',NEW_TX)]:
        code,d=request(base,'/api/simulate',dict(objects=objects,tx_positions=tx,seed=112,snr_db=30))
        check(lid+' live simulation model binding',code==200 and d['model_layout_id']==lid and d['model_position_validated']==(lid!='unvalidated'))
        if lid=='unvalidated':check('Arbitrary edits never accepted',d['status']!='Accepted regions' and d['confirmed_count']==0 and all(not r['accepted'] for r in d['occupancy']['regions']))
        simulations.append(d)
    check('Transmitter change actually changes CSI',len({r['signal_sha256'] for r in simulations[:3]})==3)
    check('Preset reset reuses correct clean CSI',simulations[0]['signal_sha256']==simulations[-1]['signal_sha256'] and simulations[-1]['channel_cached'])
    check('No weight replacement during runtime',sha(O5/'model.joblib')==read(O5/'frozen.json')['model_sha256'])
    atomic(output,dict(passed=True,completed_at=now(),base=base,checks=checks,model_sha256=sha(O5/'model.joblib'),note='Functional checks only; arbitrary source changes remain unvalidated.'))
    print(json.dumps(dict(passed=True,checks=len(checks)),indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',default='http://127.0.0.1:8769');p.add_argument('--output',default='api_verification.json');a=p.parse_args();main(a.base,a.output)
