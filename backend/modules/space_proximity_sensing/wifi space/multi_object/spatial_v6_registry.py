
from pathlib import Path
from functools import lru_cache
import os,json,hashlib,joblib,time
from multi_common import OUT,DATA
O=OUT/'spatial_v6'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def folder(lid):
 if lid not in ['default','t1_v1']:raise ValueError('Unknown V6 preset')
 return O/lid
def approved(lid):
 return _approved(lid,os.environ.get('ROOMSENSE_VERIFY_V6')=='1',int(time.time()//2))

@lru_cache(maxsize=8)
def _approved(lid,candidate,epoch):
 f=folder(lid)
 try:
  a=read(f/('CANDIDATE.json' if candidate else 'APPROVED.json'))
  for key,name in [('frozen_sha256','frozen.json'),('results_sha256','results.json')]:
   if sha(f/name)!=a[key]:return False
  frozen=read(f/'frozen.json')
  if a['model_sha256']!=frozen['model_sha256'] or not read(f/'results.json')['release_check']['eligible']:return False
  for name,digest in frozen['source_sha256'].items():
   if sha(Path(__file__).with_name(name))!=digest:return False
  if not candidate and sha(f/'api_verification.json')!=a['api_sha256']:return False
  return True
 except (OSError,KeyError,ValueError):return False
@lru_cache(maxsize=2)
def cached(lid,stamp):
 f=folder(lid)
 if sha(f/'model.joblib')!=read(f/'frozen.json')['model_sha256']:raise RuntimeError('V6 model checksum mismatch')
 return joblib.load(f/'model.joblib')
def model(lid):
 if not approved(lid):return None
 f=folder(lid);return cached(lid,(f/'model.joblib').stat().st_mtime_ns)
def evaluation(lid):return read(folder(lid)/'results.json') if approved(lid) else None
def catalog():
 if not (DATA/'spatial_v6/manifest.json').exists():return {}
 rows=read(DATA/'spatial_v6/manifest.json');result={}
 for lid in ['default','t1_v1']:
  if not approved(lid):continue
  for r in rows:
   if r['split'] not in ['test','boundary_test']:continue
   rid='v6_'+lid+'_'+r['id'][3:];result[rid]=dict(r,id=rid,raw_id=r['id'],layout_id=lid,count=len(r['objects']),region_count=len({p['zone'] for p in r['objects']}),shared=len({p['zone'] for p in r['objects']})<len(r['objects']))
 return result
