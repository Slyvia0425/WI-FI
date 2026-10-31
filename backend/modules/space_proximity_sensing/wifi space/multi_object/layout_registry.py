
from pathlib import Path
from functools import lru_cache
import json,hashlib,os,numpy as np,joblib
from multi_common import OUT,DATA,TX
NEW_TX=[p[:] for p in TX];NEW_TX[0]=[1.1,1.2,.45]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def approved_v1():
 f=OUT/'layout_t1_v1'
 try:
  marker='CANDIDATE.json' if os.environ.get('ROOMSENSE_VERIFY_T1')=='1' else 'APPROVED.json'
  a=read(f/marker);frozen=read(f/'frozen.json')
  for key,name in [('frozen_sha256','frozen.json'),('results_sha256','results.json'),('audit_sha256','data_audit_full.json'),('offline_sha256','offline_verification.json'),('evaluation_binding_sha256','evaluation_binding.json')]:
   if hashlib.sha256((f/name).read_bytes()).hexdigest()!=a[key]:return False
  if marker=='APPROVED.json' and hashlib.sha256((f/'api_verification.json').read_bytes()).hexdigest()!=a['api_sha256']:return False
  binding=read(f/'evaluation_binding.json')
  return a['model_sha256']==frozen['model_sha256']==binding['model_sha256'] and binding['passed'] and read(f/'results.json')['release_check']['eligible'] and read(f/'data_audit_full.json')['clean_split_audit']
 except (OSError,KeyError,ValueError):return False
def approved_v3():
 f=OUT/'layout_t1_v3'
 try:
  candidate=os.environ.get('ROOMSENSE_VERIFY_T1_V3')=='1';a=read(f/('CANDIDATE.json' if candidate else 'APPROVED.json'))
  for key,name in [('frozen_sha256','frozen.json'),('results_sha256','results.json'),('audit_sha256','data_audit.json'),('binding_sha256','evaluation_binding.json')]:
   if hashlib.sha256((f/name).read_bytes()).hexdigest()!=a[key]:return False
  if not candidate and hashlib.sha256((f/'api_verification.json').read_bytes()).hexdigest()!=a['api_sha256']:return False
  return a['model_sha256']==read(f/'frozen.json')['model_sha256']==read(f/'evaluation_binding.json')['model_sha256'] and read(f/'evaluation_binding.json')['passed'] and read(f/'results.json')['release_check']['eligible'] and read(f/'data_audit.json')['passed']
 except (OSError,KeyError,ValueError):return False
def approved():return approved_v3() or approved_v1()
def current_folder():return OUT/('layout_t1_v3' if approved_v3() else 'layout_t1_v1')
@lru_cache(maxsize=2)
def model(folder,stamp):
 f=Path(folder);p=f/'model.joblib'
 h=hashlib.sha256(p.read_bytes()).hexdigest()
 if h!=read(f/'frozen.json')['model_sha256']:raise RuntimeError('Preset model checksum mismatch')
 return joblib.load(p)
def match(tx):
 from spatial_v6_registry import model as v6_model
 if np.allclose(tx,NEW_TX,rtol=0,atol=1e-7):
  upgraded=v6_model('t1_v1')
  if upgraded is not None:return 't1_v1',upgraded
 if np.allclose(tx,TX,rtol=0,atol=1e-7):return 'default',v6_model('default')
 if np.allclose(tx,NEW_TX,rtol=0,atol=1e-7) and approved():
  f=current_folder();p=f/'model.joblib';return 't1_v1',model(str(f),p.stat().st_mtime_ns)
 return 'unvalidated',None
def entries():
 ready=approved();folder=current_folder();m=read(folder/'results.json')['splits']['test'] if ready else None
 es=[dict(id='default',name='默认布局 / Default layout',tx_positions=TX,ready=True,revision='default_v5'),dict(id='t1_v1',name='新位置 T1 / New T1 layout',tx_positions=NEW_TX,ready=ready,revision='t1_count_v3' if folder.name=='layout_t1_v3' else 't1_v1',evaluation=read(folder/'results.json') if ready else None,metrics=None if m is None else {k:m[k] for k in ['n','occupied_zone_macro_f1','count_accuracy','exact_joint_accuracy']})]
 from spatial_v6_registry import evaluation as v6_evaluation
 for e in es:
  upgraded=v6_evaluation(e['id'])
  if upgraded is not None:
   e.update(ready=True,revision='spatial_count_v6',evaluation=upgraded,metrics={k:upgraded['splits']['test'][k] for k in ['n','occupied_zone_macro_f1','count_accuracy','exact_joint_accuracy']})
 return es

@lru_cache(maxsize=1)
def new_catalog(path,stamp):
 rows=read(Path(path))
 return {r['id']:r for r in rows}
def catalog():
 from spatial_v6_registry import catalog as v6_catalog
 old={}
 if approved():
  p=DATA/current_folder().name/'manifest.json';old=new_catalog(str(p),p.stat().st_mtime_ns)
 return dict(old,**v6_catalog())
def by_id(id):
 from spatial_v6_registry import model as v6_model
 if id in ['default','t1_v1']:
  upgraded=v6_model(id)
  if upgraded is not None:return upgraded,TX if id=='default' else NEW_TX
 if id=='default':return None,TX
 if id=='t1_v1' and approved():return match(NEW_TX)[1],NEW_TX
 raise ValueError('Unknown or unapproved model layout')
