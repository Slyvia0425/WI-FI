
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np,joblib
from threadpoolctl import threadpool_limits
from home_v5_verify import request
from spatial_v6_infer import infer_v5
O=Path('outputs/spatial_v6')
def main(base,name):
 checks=[];code,s=request(base,'/api/state');assert code==200
 def check(n,v):
  checks.append(dict(name=n,passed=bool(v)))
  if not v:raise AssertionError(n)
 check('State ready',s['ready'] and s['model_version']==5)
 check('Default summary bound to active V6 evaluation',s['occupancy_results']['splits']['test']==json.loads((O/'default/results.json').read_text())['splits']['test'])
 code,h=request(base,'/api/ready');check('Full service ready',code==200 and h['ready'])
 for e in s['config']['available_layouts']:
  lid=e['id'];upgraded=e['revision']=='spatial_count_v6'
  rows=[r for r in s['records'] if r['layout_id']==lid];check(lid+' matched scene catalog',len(rows)==(1500 if upgraded else 3000))
  if upgraded:check(lid+' matching evaluation',e['evaluation']['splits']['test']==json.loads((O/lid/'results.json').read_text())['splits']['test'])
  code,payload=request(base,'/api/example-json?layout_id='+lid);check(lid+' signal shape',code==200 and np.shape(payload['real'])==(16,32,64))
  code,d=request(base,'/api/predict',payload);check(lid+' matched upload model',code==200 and d['model_layout_id']==lid and (d['inference_model_revision']=='spatial_count_v6')==upgraded and d['truth'] is None)
  _,again=request(base,'/api/predict',payload);check(lid+' deterministic upload',d['occupancy']['per_zone_object_count']==again['occupancy']['per_zone_object_count'])
  if upgraded:
   b=joblib.load(O/lid/'model.joblib');local=infer_v5(b,np.asarray(payload['real'])+1j*np.asarray(payload['imag']));check(lid+' local frozen inference matches API',np.allclose(local['probabilities'],d['occupancy']['probabilities'],atol=1e-10) and local['per_zone_object_count']==d['occupancy']['per_zone_object_count'] and local['status']==d['occupancy']['status'])
   check(lid+' confirmed requires expert agreement',d['occupancy']['status']!='Accepted regions' or d['occupancy']['expert_agreement'])
  row=next(r for r in rows if r['split']=='test' and r['shared'] and r['count']==3)
  code,replay=request(base,'/api/record',dict(id=row['id'],layout_id=lid,seed=3168));check(lid+' shared-three replay routing',code==200 and replay['model_layout_id']==lid and replay['true_count']==3 and (replay['inference_model_revision']=='spatial_count_v6')==upgraded)
  o=replay['occupancy'];check(lid+' count consistency',sum(o['per_zone_object_count'])==o['estimated_object_count'])
  code,_=request(base,'/api/record',dict(id=row['id'],layout_id='t1_v1' if lid=='default' else 'default'));check(lid+' mismatched replay rejected',code==400)
  zero=dict(real=np.zeros((16,32,64)).tolist(),imag=np.zeros((16,32,64)).tolist());
  if lid!='default':zero['layout_id']=lid
  code,z=request(base,'/api/predict',zero);check(lid+' zero Unknown clears regions',code==200 and z['status']=='Unknown' and not z['occupancy']['regions'] and z['estimated_count'] is None)
  single=dict(real=payload['real'][0],imag=payload['imag'][0]);
  if lid!='default':single['layout_id']=lid
  code,z=request(base,'/api/predict',single);check(lid+' single frame never confirmed',code==200 and z['status']!='Accepted regions')
  code,_=request(base,'/api/predict',dict(payload,objects=[{'zone':4}]));check(lid+' geometry hint rejected',code==400)
 code,_=request(base,'/api/record',dict(id='h5_train_00000'));check('Training replay blocked',code==400)
 result=dict(passed=True,checks=checks,base=base,at=time.time(),scope='Functional routing and invalid-input checks; independent accuracy remains in results.json')
 (O/name).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({'passed':True,'checks':len(checks)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--base',default='http://127.0.0.1:8775');p.add_argument('--output',default='api_candidate.json');a=p.parse_args()
 with threadpool_limits(limits=3):main(a.base,a.output)
