import json,urllib.request,urllib.error,numpy as np
from multi_common import TX,SIZES
from pathlib import Path
objects=[dict(kind=0,x=3.02,y=2.69,size=SIZES[0].tolist()),dict(kind=2,x=3.06,y=3.53,size=SIZES[2].tolist())]
def call(tx):
 req=urllib.request.Request('http://127.0.0.1:8769/api/simulate',data=json.dumps(dict(objects=objects,tx_positions=tx,seed=42,snr_db=30)).encode(),headers={'Content-Type':'application/json'})
 try:
  with urllib.request.urlopen(req,timeout=90) as r:return r.status,json.load(r)
 except urllib.error.HTTPError as e:return e.code,json.load(e)
a=call(TX);m=[p[:] for p in TX];m[0]=[1.45,1.2,.45];b=call(m);c=call(TX)
assert all(x[0]==200 for x in [a,b,c])
assert a[1]['signal_sha256']!=b[1]['signal_sha256']
error=float(np.max(np.abs(np.array(a[1]['link_delta_db'])-c[1]['link_delta_db'])))
assert error<.05,error
assert b[1]['occupancy']['status']=='Uncertain' and all(not r['accepted'] for r in b[1]['occupancy']['regions'])
for pos in [[[float('nan'),1,.45]]*4,[[7,1,.45]]*4,[[3.02,2.69,.45]]+TX[1:],[[3,5.72,.45]]+TX[1:],[TX[1]]+TX[1:]]:assert call(pos)[0]==400
r=dict(passed=True,checks=['fresh CSI changes with TX','default reset restores link response within0.05dB numerical tolerance','moved layout remains unvalidated/Uncertain','NaN bounds object cabinet radio collisions rejected'],reset_max_delta_db=error,default=a[1]['signal_sha256'],moved=b[1]['signal_sha256'])
Path('outputs/tx_editor/api_verification.json').write_text(json.dumps(r,indent=2),encoding='utf-8');print(r)
