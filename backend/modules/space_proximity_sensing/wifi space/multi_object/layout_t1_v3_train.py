
import os,time,json,copy,traceback,argparse
from pathlib import Path
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from layout_t1_v1_cache import load
from layout_t1_v1_common import STATE_COUNTS,sha
from layout_t1_v1_infer import batch
from layout_t1_v2_select import evaluate
O=Path('outputs/layout_t1_v3');O.mkdir(exist_ok=True)
def atomic(name,obj):
 p=O/name;t=p.with_name(p.name+'.'+str(os.getpid())+'.partial');t.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
 for i in range(40):
  try:t.replace(p);return
  except PermissionError:
   if i==39:raise
   time.sleep(.05)
def dump(b,p):
 t=p.with_suffix('.partial');joblib.dump(b,t);t.replace(p)
def fit(kind):
 path=O/(kind+'.joblib')
 if path.exists():return
 tr=load('train');va=load('validation');start=time.time();atomic(kind+'_state.json',dict(state='training',pid=os.getpid(),at=start))
 if kind=='boost':
  from catboost import CatBoostRegressor
  spec=dict(loss_function='MultiRMSE',iterations=1800,depth=7,learning_rate=.06,l2_leaf_reg=10,random_seed=2026103094,task_type='GPU',devices='0',gpu_ram_part=.3,thread_count=3,allow_writing_files=False,verbose=150)
  model=CatBoostRegressor(**spec)
  model.fit(np.asarray(tr['calibrated']),np.asarray(tr['zone_count']),sample_weight=1+.5*(tr['object_count']==3),eval_set=(np.asarray(va['calibrated']),np.asarray(va['zone_count'])),early_stopping_rounds=150,use_best_model=True)
 elif kind=='forest':
  spec=dict(n_estimators=200,max_features=.2,min_samples_leaf=2,n_jobs=2,random_state=2026103095)
  model=ExtraTreesRegressor(**spec);model.fit(np.asarray(tr['calibrated'][::2]),np.asarray(tr['zone_count'][::2]))
 else:
  spec=dict(hidden_layer_sizes=[512,256],batch_size=256,alpha=.1,learning_rate_init=.0007,random_state=3096,max_epochs=140,patience=25)
  scaler=StandardScaler().fit(tr['calibrated']);X=scaler.transform(tr['calibrated']);V=scaler.transform(va['calibrated']);Y=np.asarray(tr['zone_count'],np.float32);truth=np.asarray(va['zone_count']);m=MLPRegressor(**{k:v for k,v in spec.items() if k not in ['max_epochs','patience']});best=None;best_loss=float('inf');best_epoch=0
  for epoch in range(1,141):
   m.partial_fit(X,Y)
   if epoch%5:continue
   pred=m.predict(V);loss=float(np.mean((pred-truth)**2))
   if loss<best_loss-1e-6:best_loss=loss;best_epoch=epoch;best=copy.deepcopy(m)
   dump(dict(model=m,scaler=scaler,epoch=epoch,best=best,best_epoch=best_epoch,best_loss=best_loss),O/'neural_checkpoint.joblib')
   atomic(kind+'_state.json',dict(state='training',pid=os.getpid(),epoch=epoch,best_epoch=best_epoch,validation_mse=best_loss))
   if epoch>=50 and epoch-best_epoch>=25:break
  if hasattr(best,'_optimizer'):del best._optimizer
  model=make_pipeline(scaler,best)
 dump(model,path);atomic(kind+'_state.json',dict(state='completed',pid=os.getpid(),elapsed_s=time.time()-start,spec=spec))
def select():
 assert not (O/'frozen.json').exists(),'Experiment sealed; create a new version'
 va=load('validation');b=joblib.load('outputs/layout_t1_v1/model.joblib');original=batch(b,va);base=original['posterior'];usable=(original['routing']=='calibrated')&original['quality']
 def score(post):
  m=evaluate(post,va);single=va['object_count']==1;m['single_joint']=float(np.mean(np.all(STATE_COUNTS[post.argmax(1)][single]==va['zone_count'][single],axis=1)));return m
 baseline=score(base);pred={}
 for k in ['boost','forest','neural']:
  if (O/(k+'.joblib')).exists():pred[k]=joblib.load(O/(k+'.joblib')).predict(va['calibrated'])
 if 'boost' in pred and 'forest' in pred:pred['mixed']=(pred['boost']+pred['forest'])/2
 if 'neural' in pred:
  for k in ['boost','forest']:
   if k in pred:pred[k+'_neural']=(pred[k]+pred['neural'])/2
 rows=[]
 for name,z in pred.items():
  for scale in [.9,1.,1.1,1.2]:

   error=((z[:,None,:]*scale-STATE_COUNTS[None,:,:])**2).sum(2)
   for weight in [.25,.5,1.,2.,4.,8.,16.]:
    logits=np.log(np.maximum(base,1e-30))-weight*error;logits-=logits.max(1,keepdims=True);post=np.exp(logits);post/=post.sum(1,keepdims=True);post[~usable]=base[~usable]
    rows.append(dict(regressor=name,scale=scale,weight=weight,**score(post)))
 candidates=[r for r in rows if r['zone_macro_f1']>=baseline['zone_macro_f1']-.005 and r['count_accuracy']>=baseline['count_accuracy']-.005 and r['three_joint']>=baseline['three_joint'] and r['single_joint']>=baseline['single_joint'] and r['balanced_joint']>=baseline['balanced_joint']]
 best=max(candidates,key=lambda r:(r['balanced_joint'],r['exact_joint_accuracy'])) if candidates else None
 accepted=best is not None and best['exact_joint_accuracy']>=baseline['exact_joint_accuracy']+.01
 atomic('selection.json',dict(baseline=baseline,best=best,candidates=rows,accepted_for_fresh_evaluation=accepted,base_model_sha256=sha(Path('outputs/layout_t1_v1/model.joblib')),scope='Training and original validation only; previous final tests not loaded. Minimum validation joint gain1pp. Fresh paired evaluation required.'));print(json.dumps(dict(baseline=baseline,best=best,accepted=accepted),indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['boost','forest','neural','select']);a=p.parse_args()
 try:
  with threadpool_limits(limits=3):select() if a.stage=='select' else fit(a.stage)
 except Exception as e:atomic(a.stage+'_state.json',dict(state='failed',error=str(e),traceback=traceback.format_exc()));raise
