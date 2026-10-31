
import os,time,copy,json,hashlib,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import joblib,numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from layout_t1_v1_common import O5,D5,read,atomic,lock,alive,now,decode,state_ids,STATE_COUNTS
from layout_t1_v1_cache import load
from opt_v2_common import MASKS,layout_ids
from night_infer import component_prob
SPEC=dict(version=5,max_epochs=180,min_epochs=50,patience=30,every=5,alpha=.3,learning_rate=.0007,trees=3000,depth=8,tree_lr=.04)

def dump(b,p):
    tmp=p.with_suffix('.'+str(os.getpid())+'.partial');joblib.dump(b,tmp);tmp.replace(p)
def group_score(y,counts,pred,nobj,true_zone_counts=None,pred_zone_counts=None):
    exact=np.all(y==pred,axis=1)&(counts==nobj)
    if true_zone_counts is not None:exact&=np.all(true_zone_counts==pred_zone_counts,axis=1)
    shared=counts>y.sum(1);scores=[]
    for s in [False,True]:
        for n in [2,3]:
            ix=(shared==s)&(counts==n)
            if ix.any():scores.append(exact[ix].mean())
    return float(np.mean(scores))

def fit(name):
    view,kind=name.rsplit('_',1)
    with lock('fit_'+name):
        tr=load('train',True);va=load('validation',True);folder=O5/'components'/name;folder.mkdir(exist_ok=True,parents=True);path=folder/'model.joblib'
        meta=dict(name=name,train=tr['meta']['fingerprint'],validation=va['meta']['fingerprint'],spec=SPEC,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());digest=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
        if path.exists() and (folder/'complete.json').exists():assert read(folder/'complete.json')['fingerprint']==digest;return dict(name=name,path=str(path))
        (folder/'spec.json').write_text(json.dumps(dict(meta,fingerprint=digest),indent=2),encoding='utf-8');start=time.time()
        if kind=='tree':
            from catboost import CatBoostClassifier
            while (O5/'gpu_data.lock').exists() and alive(read(O5/'gpu_data.lock')['pid']):time.sleep(3)
            with lock('gpu_train'):
                atomic('fit_'+name+'_state.json',dict(state='training_gpu',pid=os.getpid(),updated=now()))
                m=CatBoostClassifier(loss_function='MultiLogloss',iterations=SPEC['trees'],depth=SPEC['depth'],learning_rate=SPEC['tree_lr'],l2_leaf_reg=10,random_seed=2026103072,task_type='GPU',devices='0',gpu_ram_part=.4,thread_count=4,allow_writing_files=False,verbose=200)
                m.fit(np.asarray(tr[view][::2]),np.asarray(tr['labels'][::2]),eval_set=(np.asarray(va[view]),np.asarray(va['labels'])),early_stopping_rounds=200,use_best_model=True)
                b=dict(model=m,kind='tree',view=view,fingerprint=digest)
        else:
            scaler=StandardScaler().fit(tr[view]);X=scaler.transform(tr[view]);V=scaler.transform(va[view]);counts=tr['object_count'];vy=va['labels'];vc=va['object_count']
            y=state_ids(tr['zone_count']) if kind=='layout' else counts-1 if kind=='objects' else tr['labels'].sum(1).astype(int)-1
            target=state_ids(va['zone_count']) if kind=='layout' else vc-1 if kind=='objects' else vy.sum(1).astype(int)-1
            classes=np.arange(len(STATE_COUNTS) if kind=='layout' else 3);hidden=(512,256) if kind=='layout' else (256,128) if kind=='objects' else (192,96);m=MLPClassifier(hidden_layer_sizes=hidden,batch_size=256,alpha=SPEC['alpha'],learning_rate_init=SPEC['learning_rate'],random_state=154+['regions','objects','layout'].index(kind),early_stopping=False)
            ck=folder/'checkpoint.joblib';start_epoch=1;best_score=-1.;best_epoch=0;best=None;trace=[]
            if ck.exists():
                old=joblib.load(ck);assert old['fingerprint']==digest;m=old['current'];best=old['best'];best_score=old['best_score'];best_epoch=old['best_epoch'];trace=old['trace'];start_epoch=old['epoch']+1
            with threadpool_limits(limits=2):
                for epoch in range(start_epoch,SPEC['max_epochs']+1):
                    m.partial_fit(X,y,classes=classes)
                    if epoch%SPEC['every']:continue
                    p=m.predict_proba(V);pred=p.argmax(1)
                    score=float(np.mean(pred==target))
                    trace.append(dict(epoch=epoch,score=score))
                    if score>best_score+1e-6:best_score=score;best_epoch=epoch;best=copy.deepcopy(m)
                    dump(dict(fingerprint=digest,current=m,best=best,best_score=best_score,best_epoch=best_epoch,epoch=epoch,trace=trace),ck)
                    atomic('fit_'+name+'_state.json',dict(state='training',pid=os.getpid(),updated=now(),epoch=epoch,best_epoch=best_epoch,best_score=best_score));print(name,epoch,score,flush=True)
                    if epoch>=SPEC['min_epochs'] and epoch-best_epoch>=SPEC['patience']:break
            if hasattr(best,'_optimizer'):del best._optimizer
            b=dict(model=best,scaler=scaler,kind=kind,view=view,fingerprint=digest,validation_score=best_score,best_epoch=best_epoch);(folder/'trace.json').write_text(json.dumps(trace,indent=2),encoding='utf-8')
        dump(b,path);(folder/'complete.json').write_text(json.dumps(dict(fingerprint=digest,completed_at=now(),elapsed_s=time.time()-start),indent=2),encoding='utf-8');atomic('fit_'+name+'_state.json',dict(state='completed',pid=os.getpid(),updated=now(),path=str(path)));return dict(name=name,path=str(path))

def select(results):
    va=load('validation');y=va['labels'];counts=va['object_count'];reference=np.load(D5/'empty_reference.npy');experts={};reports={}
    with threadpool_limits(limits=4):
        for view in ['calibrated','phase_free']:
            components={k:joblib.load(next(r['path'] for r in results if r['name']==view+'_'+k)) for k in ['tree','regions','objects','layout']};X=va[view]
            p=component_prob(components['tree'],X);rc=component_prob(components['regions'],X);oc=component_prob(components['objects'],X);lp=component_prob(components['layout'],X);best=None;comparisons=[]
            for w in [1.]:
                mix=p
                for bias in [-.25,0,.25]:
                    for rw in [0,.5,1]:
                        for lw in [.25,.5,1]:
                            for ow in [.5,1,2]:
                                cfg=dict(tree_weight=w,odds_bias=bias,region_weight=rw,layout_weight=lw,object_weight=ow);pred,nobj,post=decode(mix,rc,lp,oc,cfg);gs=group_score(y,counts,pred,nobj,va['zone_count'],STATE_COUNTS[post.argmax(1)]);f1=float(f1_score(y,pred,average='macro',zero_division=0));score=(f1>=.85 and float(np.mean(nobj==counts))>=.85,gs,f1,float(np.mean(nobj==counts)),w);m=dict(config=cfg,balanced_joint_exact=gs,zone_macro_f1=f1,object_count_accuracy=score[3],validation_target_met=bool(score[0]));comparisons.append(m)
                                if best is None or score>best[0]:best=(score,m)
            cfg=best[1]['config'];e=dict(components=components,decoder=cfg)
            experts[view]=e;reports[view]=dict(best=best[1],comparison=sorted(comparisons,key=lambda a:(a['balanced_joint_exact'],a['zone_macro_f1'],a['object_count_accuracy']),reverse=True)[:8])
    primary=max(experts,key=lambda v:(reports[v]['best']['validation_target_met'],reports[v]['best']['balanced_joint_exact']));candidate=dict(version=5,name='Furnished-room joint occupancy and physical object count',reference=reference,frames=16,primary_view=primary,experts=experts,selection_reports=reports,train_cache=load('train')['meta']['fingerprint'],validation_cache=va['meta']['fingerprint'])
    dump(candidate,O5/'selected.joblib');atomic('selection.json',dict(selected_at=now(),primary=primary,reports=reports,joint_states=len(STATE_COUNTS),test_used=False));print('Shared-zone candidate selected. Calibration/final test pending.',flush=True)

def main():
    with lock('training'):
        if (O5/'selection.json').exists():return
        atomic('training_state.json',dict(state='waiting_for_cache',pid=os.getpid(),updated=now()));load('train',True);load('validation',True);atomic('training_state.json',dict(state='training',pid=os.getpid(),updated=now()))
        results=[]
        with ProcessPoolExecutor(max_workers=3) as pool:
            fs=[pool.submit(fit,v+'_'+k) for k in ['regions','objects','layout'] for v in ['calibrated','phase_free']]
            for v in ['calibrated','phase_free']:results.append(fit(v+'_tree'))
            results.extend(f.result() for f in fs)
        atomic('components.json',dict(components=results));select(results);atomic('training_state.json',dict(state='selected',pid=os.getpid(),updated=now()))
if __name__=='__main__':
    try:main()
    except Exception as e:atomic('training_state.json',dict(state='failed',pid=os.getpid(),updated=now(),error=str(e),traceback=traceback.format_exc()));raise
