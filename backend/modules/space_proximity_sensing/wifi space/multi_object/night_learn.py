
import os,json,time,copy,hashlib,traceback,warnings
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime
import joblib,numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from night_cache import load
from night_common import NIGHT,atomic,stage_lock,alive,now,log,DEADLINE,TZ
from opt_v2_common import layout_ids,MASKS
from opt_v2_train import balanced_exact
warnings.filterwarnings('ignore',category=ConvergenceWarning)
TRAIN_SPEC=dict(version=1,batch_size=256,learning_rate=.0007,alpha=.3,max_epochs=160,min_epochs=40,patience=30,evaluate_every=5,trees=2400,tree_depth=8,tree_learning_rate=.04,tree_l2=10)

def safe_dump(obj,path):
    tmp=path.with_suffix('.'+str(os.getpid())+'.partial');joblib.dump(obj,tmp);tmp.replace(path)
def json_file(path,data):
    tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)
def model_path(name,tr,va):
    spec=dict(component=name,train=tr['meta']['fingerprint'],validation=va['meta']['fingerprint'],training_spec=TRAIN_SPEC,learner_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    digest=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest();folder=NIGHT/'models'/(name+'-'+digest[:12]);folder.mkdir(exist_ok=True,parents=True);json_file(folder/'spec.json',dict(spec,fingerprint=digest));return folder,digest

@contextmanager
def gpu_claim():
    while True:
        if datetime.now(TZ)>=DEADLINE:raise RuntimeError('Night deadline before GPU allocation')
        ds=NIGHT/'gpu_dataset.lock'
        if ds.exists():
            try:busy=alive(json.loads(ds.read_text())['pid'])
            except (OSError,ValueError,KeyError):busy=False
            if busy:time.sleep(3);continue
        try:
            manager=stage_lock('gpu_training');manager.__enter__();break
        except (FileExistsError,RuntimeError):time.sleep(3)
    try:yield
    finally:manager.__exit__(None,None,None)

def fit_component(name):
    view,kind=name.rsplit('_',1)
    if view not in ['calibrated','phase_free'] or kind not in ['occ','count','layout','tree']:raise ValueError(name)
    with stage_lock('learn_'+name):
        atomic('learn_'+name+'_state.json',dict(state='waiting_for_cache',pid=os.getpid(),updated=now()))
        tr=load('train',view,wait=True);va=load('validation',view,wait=True);folder,digest=model_path(name,tr,va)
        path=folder/'model.joblib'
        if path.exists() and (folder/'complete.json').exists():
            complete=json.loads((folder/'complete.json').read_text());assert complete['fingerprint']==digest
            atomic('learn_'+name+'_state.json',dict(state='completed',pid=os.getpid(),updated=now(),path=str(path),fingerprint=digest,reused=True));return dict(name=name,path=str(path),fingerprint=digest)
        if kind=='tree':fit_tree(name,view,tr,va,folder,digest)
        else:fit_mlp(name,view,kind,tr,va,folder,digest)
        return dict(name=name,path=str(path),fingerprint=digest)

def fit_mlp(name,view,kind,tr,va,folder,digest):
    started=time.time();hidden=(384,192) if kind=='occ' else (192,96) if kind=='count' else (512,256)
    seed=173 if kind=='occ' else 82 if kind=='count' else 83
    scaler=StandardScaler().fit(tr[view]);X=scaler.transform(tr[view]);V=scaler.transform(va[view]);y=tr['labels'];vy=va['labels']
    target=y if kind=='occ' else y.sum(1).astype(int)-1 if kind=='count' else layout_ids(y)
    classes=np.arange(9 if kind=='occ' else 3 if kind=='count' else 129)
    ck=folder/'checkpoint.joblib';trace=[];best_score=-1.;best_epoch=0;best=None;start_epoch=1
    if ck.exists():
        saved=joblib.load(ck);assert saved['fingerprint']==digest;model=saved['model'];best=saved['best'];best_score=saved['best_score'];best_epoch=saved['best_epoch'];trace=saved['trace'];start_epoch=saved['epoch']+1
    else:model=MLPClassifier(hidden_layer_sizes=hidden,batch_size=TRAIN_SPEC['batch_size'],alpha=TRAIN_SPEC['alpha'],learning_rate_init=TRAIN_SPEC['learning_rate'],early_stopping=False,random_state=seed,shuffle=True)
    with threadpool_limits(limits=2):
        for epoch in range(start_epoch,TRAIN_SPEC['max_epochs']+1):
            if datetime.now(TZ)>=DEADLINE:raise RuntimeError('Deadline; MLP checkpoint retained')
            model.partial_fit(X,target,classes=classes)
            if epoch%TRAIN_SPEC['evaluate_every']:continue
            p=model.predict_proba(V)
            score=float(f1_score(vy,p>=.4,average='macro',zero_division=0)) if kind=='occ' else float(np.mean(p.argmax(1)==vy.sum(1)-1)) if kind=='count' else balanced_exact(vy,MASKS[p.argmax(1)])
            trace.append(dict(epoch=epoch,score=score,loss=float(model.loss_)))
            if score>best_score+1e-6:best_score=score;best_epoch=epoch;best=copy.deepcopy(model)
            safe_dump(dict(fingerprint=digest,model=model,best=best,best_score=best_score,best_epoch=best_epoch,epoch=epoch,trace=trace),ck)
            atomic('learn_'+name+'_state.json',dict(state='training',pid=os.getpid(),updated=now(),epoch=epoch,best_epoch=best_epoch,best_score=best_score,score=score,elapsed_s=time.time()-started));print(name,epoch,score,flush=True)
            if epoch>=TRAIN_SPEC['min_epochs'] and epoch-best_epoch>=TRAIN_SPEC['patience']:break
    if best is None:raise RuntimeError('No selected MLP checkpoint')

    if hasattr(best,'_optimizer'):del best._optimizer
    bundle=dict(model=best,scaler=scaler,view=view,kind=kind,feature_dimension=X.shape[1],fingerprint=digest,train_cache=tr['meta']['fingerprint'],validation_cache=va['meta']['fingerprint'],best_epoch=best_epoch,validation_score=best_score,physical_training_scenes=tr['meta']['physical_scenes'],training_observations=len(X))
    safe_dump(bundle,folder/'model.joblib');json_file(folder/'trace.json',trace);finish(name,folder,digest,started,best_score)

def fit_tree(name,view,tr,va,folder,digest):
    from catboost import CatBoostClassifier
    started=time.time();X=np.asarray(tr[view][::2]);y=np.asarray(tr['labels'][::2]);V=np.asarray(va[view]);vy=np.asarray(va['labels']);errors=[];model=None

    for device in ['GPU','CPU']:
        try:
            atomic('learn_'+name+'_state.json',dict(state='waiting_for_gpu' if device=='GPU' else 'training',pid=os.getpid(),updated=now(),device=device))
            def run():
                m=CatBoostClassifier(loss_function='MultiLogloss',iterations=TRAIN_SPEC['trees'] if device=='GPU' else 1400,depth=TRAIN_SPEC['tree_depth'],learning_rate=TRAIN_SPEC['tree_learning_rate'],l2_leaf_reg=TRAIN_SPEC['tree_l2'],random_seed=2026103004,task_type=device,thread_count=4,allow_writing_files=False,verbose=200,**(dict(gpu_ram_part=.48,devices='0') if device=='GPU' else {}))
                atomic('learn_'+name+'_state.json',dict(state='training',pid=os.getpid(),updated=now(),device=device,physical_scenes=len(X)))
                m.fit(X,y,eval_set=(V,vy),early_stopping_rounds=180,use_best_model=True);return m
            if device=='GPU':
                with gpu_claim():model=run()
            else:model=run()
            break
        except Exception as e:errors.append(dict(device=device,error=str(e)));log(name+' '+device+' failed: '+str(e)[:200])
    if model is None:raise RuntimeError('Both tree backends failed')
    p=np.asarray(model.predict_proba(V));score=float(f1_score(vy,p>=.4,average='macro',zero_division=0))
    safe_dump(dict(model=model,view=view,kind='tree',feature_dimension=X.shape[1],fingerprint=digest,train_cache=tr['meta']['fingerprint'],validation_cache=va['meta']['fingerprint'],physical_training_scenes=len(X),training_observations=len(X),backend=device,backend_errors=errors,validation_score=score),folder/'model.joblib');finish(name,folder,digest,started,score)

def finish(name,folder,digest,started,score):
    hh=hashlib.sha256((folder/'model.joblib').read_bytes()).hexdigest();json_file(folder/'complete.json',dict(fingerprint=digest,model_sha256=hh,finished=now(),validation_component_score=score,elapsed_s=time.time()-started))
    atomic('learn_'+name+'_state.json',dict(state='completed',pid=os.getpid(),updated=now(),path=str(folder/'model.joblib'),fingerprint=digest,validation_component_score=score,elapsed_s=time.time()-started));log(name+' component complete, validation score='+str(score))

def predict(component,X):
    return np.asarray(component['model'].predict_proba(component['scaler'].transform(X) if 'scaler' in component else X))

def task(name):
    try:return fit_component(name)
    except Exception as e:
        atomic('learn_'+name+'_state.json',dict(state='failed',pid=os.getpid(),updated=now(),error=str(e),traceback=traceback.format_exc()));return dict(name=name,error=str(e))
