
import numpy as np,joblib
from threadpoolctl import threadpool_limits
from home_v5_common import D5,O5,read,atomic,now,sha,labels,zone_counts
from home_v5_window import window
from home_v5_infer import batch,infer_v5
from home_v5_evaluate import metrics
from night_features import views
from night_cache import QUALITY
from multi_train import seed

def main():
 assert (O5/'test_opened.json').exists(),'Run frozen final evaluation first'
 b=joblib.load(O5/'model.joblib');assert sha(O5/'model.joblib')==read(O5/'frozen.json')['model_sha256']
 rows=[r for r in read(D5/'manifest.json') if r['split']=='test'][:300]
 yy,counts=labels(rows);zc=zone_counts(rows);raw=[]
 for r in rows:
  with np.load(D5/'records'/(r['id']+'.npz'),allow_pickle=False) as a:raw.append(a['h'])
 result=dict(at=now(),scope='Same300 independently held-out geometries, post-freeze diagnostics only. Conditions reuse scenes; never count as1200 independent geometries.',conditions={})
 with threadpool_limits(limits=4):
  for condition,snr,phase in [('standard30dB',30,0),('snr20dB',20,0),('snr10dB',10,0),('common_phase_1rad',30,1.)]:
   d=dict(labels=yy,object_count=counts,zone_count=zc);vs=[]
   for row,h in zip(rows,raw):
    obs=window(h,np.random.default_rng(seed('home-v5-stress:'+row['id'])),snr)*np.exp(1j*phase);vs.append(views(obs,b['reference']))
   d.update({v:np.array([a[v] for a in vs]) for v in ['calibrated','phase_free']});d['quality']=np.array([[a[k] for k in QUALITY] for a in vs]);result['conditions'][condition]=metrics(d,batch(b,d))
  missing=0
  for row,h in zip(rows,raw):
   obs=window(h,np.random.default_rng(seed('home-v5-missing:'+row['id'])),30);obs[:,:16]=0;o=infer_v5(b,obs);missing+=o['status']=='Unknown' and o['reason']=='missing_required_links'
 result['missing_links']=dict(n=len(rows),unknown=missing,scope='16 of32 required channels missing; reject unsupported input, not evidence of reduced-device localization.')
 atomic('stress_results.json',result);print({k:{m:r[m] for m in ['n','occupied_zone_macro_f1','count_accuracy','quality_coverage','accepted_layout_coverage']} for k,r in result['conditions'].items()});print(result['missing_links'])
if __name__=='__main__':main()
