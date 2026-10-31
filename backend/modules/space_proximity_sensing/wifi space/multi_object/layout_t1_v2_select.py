
import json,time
from pathlib import Path
import joblib,numpy as np
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from layout_t1_v1_cache import load
from layout_t1_v1_infer import expert,batch
from layout_t1_v1_common import STATE_COUNTS,STATE_MASK,STATE_COUNT,sha
from layout_t1_v1_train import group_score
from night_infer import posterior,component_prob
from layout_t1_v1_common import decode
from itertools import product

OUT=Path('outputs/layout_t1_v2')
def evaluate(post,d):
 z=STATE_COUNTS[post.argmax(1)];p=z>0;c=z.sum(1)
 return dict(zone_macro_f1=float(f1_score(d['labels'],p,average='macro',zero_division=0)),count_accuracy=float(np.mean(c==d['object_count'])),exact_joint_accuracy=float(np.mean(np.all(z==d['zone_count'],axis=1))),balanced_joint=group_score(d['labels'],d['object_count'],p,c,d['zone_count'],z),three_joint=float(np.mean(np.all(z[d['object_count']==3]==d['zone_count'][d['object_count']==3],axis=1))))

def main():
 OUT.mkdir(exist_ok=True);b=joblib.load('outputs/layout_t1_v1/model.joblib');d=load('validation');parts={}
 with threadpool_limits(limits=4):
  for view,e in b['experts'].items():parts[view]=posterior(expert(b,view,d[view])[2],e['temperature'])
  base=evaluate(batch(b,d)['posterior'],d);rows=[]
  for method in ['arithmetic','geometric']:
   for weight in [0.,.1,.25,.5,.75,.9,1.]:
    a=parts['calibrated'];f=parts['phase_free'];p=weight*a+(1-weight)*f if method=='arithmetic' else posterior(np.exp(weight*np.log(np.maximum(a,1e-30))+(1-weight)*np.log(np.maximum(f,1e-30))),1.)
    rows.append(dict(method=method,weight=weight,**evaluate(p,d)))

  cp={v:{k:component_prob(e['components'][k],d[v]) for k in ['tree','regions','objects','layout']} for v,e in b['experts'].items()}
  for method,weight in product(['component_arithmetic','component_geometric'],[.5,.75,.9,1.]):
   mixed={}
   for k in cp['calibrated']:
    a=cp['calibrated'][k];f=cp['phase_free'][k]
    if method.endswith('arithmetic'):x=weight*a+(1-weight)*f
    elif k=='tree':
     z=weight*np.log(np.clip(a,1e-7,1-1e-7)/(1-np.clip(a,1e-7,1-1e-7)))+(1-weight)*np.log(np.clip(f,1e-7,1-1e-7)/(1-np.clip(f,1e-7,1-1e-7)));x=1/(1+np.exp(-z))
    else:x=np.exp(weight*np.log(np.maximum(a,1e-30))+(1-weight)*np.log(np.maximum(f,1e-30)));x/=x.sum(1,keepdims=True)
    mixed[k]=x
   for bias,rw,lw,ow in product([-.25,0,.25],[0,.5,1],[.25,.5,1],[.5,1,2]):
    cfg=dict(odds_bias=bias,region_weight=rw,layout_weight=lw,object_weight=ow)
    post=decode(mixed['tree'],mixed['regions'],mixed['layout'],mixed['objects'],cfg)[2]
    rows.append(dict(method=method,weight=weight,decoder=cfg,**evaluate(post,d)))
  candidates=[r for r in rows if r['zone_macro_f1']>=base['zone_macro_f1']-.005 and r['count_accuracy']>=base['count_accuracy']-.005 and r['three_joint']>=base['three_joint']]
  best=max(candidates,key=lambda r:(r['balanced_joint'],r['exact_joint_accuracy']))
  accepted=best['exact_joint_accuracy']>=base['exact_joint_accuracy']+.005 and best['balanced_joint']>base['balanced_joint']
  report=dict(base_model_sha256=sha(Path('outputs/layout_t1_v1/model.joblib')),scope='Validation only. Existing final tests not loaded; do not tune on future independent tests.',baseline=base,best=best,accepted_for_fresh_evaluation=accepted,candidates=rows)
  (OUT/'selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k!='candidates'},indent=2))
if __name__=='__main__':main()
