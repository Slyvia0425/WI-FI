from pathlib import Path
import json,copy,collections
import numpy as np,joblib
from threadpoolctl import threadpool_limits
from home_v5_physics import build,channel
from multi_simulate import place
from tx_simulation import check
from multi_common import TX
from home_v5_window import window
from spatial_v6_infer import infer_v5
O=Path('outputs/failure_diagnosis_20261031');base=json.loads((O/'results.json').read_text())['scene'];b=joblib.load('outputs/spatial_v6/default/model.joblib');scene,objects,solver=build()
configs=[('unchanged',None,None),('large_inside_Z4',(1.,3.),None),('small_inside_Z3',None,(4.4,.45)),('both_inside',(1.,3.),(4.4,.45)),('large_inside_lower',(1.,2.7),None),('large_inside_upper',(1.,3.3),None),('large_crosses_x_only',(1.6,3.1),None)]
rows=[]
with threadpool_limits(limits=3):
 for name,lxy,sxy in configs:
  ps=copy.deepcopy(base)
  for o in ps:
   if o['kind']==2 and lxy:o['x'],o['y']=lxy
   if o['kind']==0 and sxy:o['x'],o['y']=sxy
  try:check(ps,TX)
  except ValueError as e:rows.append(dict(case=name,error=str(e)));continue
  place(objects,ps);h=channel(scene,solver);pred=[]
  for seed in range(6000,6010):
   d=infer_v5(b,window(h,np.random.default_rng(seed),30));pred.append(dict(pred=d.get('per_zone_object_count'),status=d['status'],correct=d.get('per_zone_object_count')==[0,0,2,1,0,0,0,0,0]))
  rows.append(dict(case=name,objects=ps,correct=sum(r['correct'] for r in pred),n=10,predictions=pred))
(O/'boundary_interventions.json').write_text(json.dumps(dict(rows=rows,note='Prespecified coordinate interventions keep true center-zone count labels identical. Case-study diagnostics only; no fitting or model changes.'),indent=2),encoding='utf-8')
for r in rows:print(r['case'],r.get('correct'),r.get('n'),r.get('error'),dict(collections.Counter(str(a['pred']) for a in r.get('predictions',[]))))
