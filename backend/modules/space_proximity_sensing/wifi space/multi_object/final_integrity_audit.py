
from pathlib import Path
import json,hashlib,time,collections,sys,ast
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'/'final_audit_20261031';OUT.mkdir(parents=True,exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
checks=[];provenance={}
def check(name,ok,**detail):
 checks.append(dict(name=name,passed=bool(ok),**detail))
 if not ok:print('FAIL',name,flush=True)
for folder in ['home_v5','layout_t1_v1','layout_t1_v3']:
 f=ROOT/'outputs'/folder;frozen=read(f/'frozen.json')
 check(folder+' frozen model',sha(f/'model.joblib')==frozen['model_sha256'])
 for name,expected in frozen.get('source_sha256',{}).items():check(folder+' frozen source '+name,sha(ROOT/name)==expected)
 if 'selection_sha256' in frozen:check(folder+' selection',sha(f/'selection.json')==frozen['selection_sha256'])
from layout_registry import approved_v1,approved_v3,entries
check('V1 approval bindings',approved_v1());check('V3 approval bindings',approved_v3())
check('V3 deployed',next(x for x in entries() if x['id']=='t1_v1')['revision']=='t1_count_v3')
from home_v5_physics import check as physical_default
from layout_t1_v1_physics import check as physical_t1
ids_seen={};all_t1_ids=set();splits_by_layout={};total=0
for name in ['home_v5','layout_t1_v1','layout_t1_v2','layout_t1_v3']:
 d=ROOT/'data'/name;manifest=d/'manifest.json';rows=read(manifest);seen=set();errors=[];aggregate=hashlib.sha256();splits=collections.Counter();groups={};geometry={}
 for i,r in enumerate(rows):
  rid=r['id'];p=d/'records'/(rid+'.npz')
  try:
   assert rid not in seen,'duplicate ID';seen.add(rid);splits[r['split']]+=1
   group=r.get('group_id',rid);assert groups.setdefault(group,r['split'])==r['split'],'group split leakage'
   geom=json.dumps(r['objects'],sort_keys=True);assert geometry.setdefault(geom,r['split'])==r['split'],'geometry split leakage'
   (physical_default if name=='home_v5' else physical_t1)(r['objects'])
   raw=p.read_bytes();aggregate.update(rid.encode()+b'\0'+hashlib.sha256(raw).digest())
   with np.load(p,allow_pickle=False) as a:
    h=a['h'];assert h.shape==(32,64) and np.iscomplexobj(h) and np.isfinite(h).all() and np.max(np.abs(h))>0,'bad CSI'
  except Exception as e:errors.append({'id':rid,'error':str(e)})
  if (i+1)%10000==0:print(name,i+1,flush=True)
 check(name+' all metadata/raw files',not errors,n=len(rows),splits=dict(splits),errors=errors[:20])
 if name!='home_v5':
  check(name+' no reused T1 IDs',not(seen&all_t1_ids));all_t1_ids.update(seen)
 provenance[str(manifest.relative_to(ROOT))]=sha(manifest)
 provenance[str(d.relative_to(ROOT))+'/records (ordered ID + file SHA256 aggregate)']=aggregate.hexdigest()
 total+=len(rows)
 print(name,'done',len(rows),'errors',len(errors),flush=True)
for p in sorted((ROOT/'outputs'/'layout_t1_v3').glob('*.npz')):provenance[str(p.relative_to(ROOT))]=sha(p)
for name in ['home_v5_window.py','occupancy.py','night_common.py','layout_t1_v1_common.py','multi_common.py','layout_registry.py','app.py','launch.py','web/app.js','web/index.html','web/style.css']:
 p=ROOT/name
 if p.exists():provenance[name]=sha(p)
for p in ROOT.glob('*.py'):
 try:ast.parse(p.read_text(encoding='utf-8-sig'));ok=True
 except Exception:ok=False
 check('Python syntax '+p.name,ok)
report=dict(passed=all(x['passed'] for x in checks),completed_at=time.time(),raw_scenes_checked=total,checks=checks,provenance=provenance,note='Supplementary hashes recorded after training; they are not retroactive preregistration. Approved artifacts and model parameters unchanged.')
(OUT/'integrity.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'passed':report['passed'],'raw_scenes_checked':total,'checks':len(checks)}),flush=True)
sys.exit(0 if report['passed'] else 1)
