from pathlib import Path
import json,hashlib,time
import numpy as np
from spatial_v6_registry import sha,approved
O=Path('outputs/spatial_v6');D=Path('data/spatial_v6');rows=json.loads((D/'manifest.json').read_text());seen={};digest=hashlib.sha256();errors=[]
for r in rows:
 p=D/'default/records'/(r['id']+'.npz')
 with np.load(p) as a:h=a['h']
 assert h.shape==(32,64) and np.isfinite(h).all() and np.linalg.norm(h)>0
 hh=hashlib.sha256(h.tobytes()).hexdigest();assert hh not in seen or seen[hh]==r['split'];seen[hh]=r['split'];digest.update(r['id'].encode()+bytes.fromhex(hh))
old_overlap=0
old=json.loads(Path('data/home_v5/manifest.json').read_text())
for r in old:
 with np.load(Path('data/home_v5/records')/(r['id']+'.npz')) as a:hh=hashlib.sha256(a['h'].tobytes()).hexdigest()
 old_overlap+=hh in seen
assert old_overlap==0
checks=dict(raw_scenes=len(rows),all_metadata_files=True,finite_complex_shapes=True,cross_split_channel_overlap=0,old_default_scenes_checked=len(old),old_channel_overlap=old_overlap,ordered_raw_channel_fingerprint=digest.hexdigest(),manifest_sha256=sha(D/'manifest.json'),geometry_audit=json.loads((O/'data_audit.json').read_text()))
(O/'full_data_audit.json').write_text(json.dumps(dict(passed=True,at=time.time(),**checks),indent=2),encoding='utf-8')
f=O/'default';a=json.loads((f/'APPROVED.json').read_text());a.update(full_data_audit_sha256=sha(O/'full_data_audit.json'),runtime_sources={n:sha(Path(n)) for n in ['app.py','layout_registry.py','spatial_v6_registry.py','web/app.js']});(f/'APPROVED.json').write_text(json.dumps(a,indent=2),encoding='utf-8')
print(json.dumps({'passed':True,'raw_scenes':len(rows),'old_channels_compared':len(old),'overlap':old_overlap}))
