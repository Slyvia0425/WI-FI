
import os,time,json,traceback
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from night_common import NIGHT,atomic,stage_lock,log,now
from opt_v2_common import OUT2,decode_scores
from opt_v2_train import raw_prob,balanced_exact,metrics

def adjusted(p,bias):
    p=np.clip(p,1e-7,1-1e-7);return 1/(1+np.exp(-(np.log(p/(1-p))+bias)))

def main():
    with stage_lock('catboost_pilot'):
        if (NIGHT/'catboost_pilot.json').exists():log('CatBoost pilot already finished; reuse its report.');return
        started=time.time();atomic('catboost_state.json',dict(state='loading',pid=os.getpid(),started=now(),scope='No test/calibration files opened'))
        from catboost import CatBoostClassifier
        with np.load(OUT2/'features_train.npz') as a:X=a['X'][::2];y=a['y'][::2]
        with np.load(OUT2/'features_validation.npz') as a:V=a['X'];vy=a['y']
        b=joblib.load(OUT2/'model.joblib');errors=[];model=None;device=None
        with threadpool_limits(limits=4):p,c,l=raw_prob(b,V)
        base_pred,_,_=decode_scores(adjusted(p,b['odds_bias']),c,l,b['count_weight'],b['layout_weight']);baseline=metrics(vy,base_pred)
        for device in ['GPU','CPU']:
            try:
                atomic('catboost_state.json',dict(state='training',pid=os.getpid(),device=device,started=now(),train_scenes=len(X),validation_scenes=len(V)))
                model=CatBoostClassifier(loss_function='MultiLogloss',iterations=1200 if device=='GPU' else 700,depth=7,learning_rate=.045,l2_leaf_reg=8,random_seed=2026103003,task_type=device,thread_count=8,allow_writing_files=False,verbose=100,**(dict(gpu_ram_part=.5,devices='0') if device=='GPU' else {}))
                model.fit(X,y,eval_set=(V,vy),early_stopping_rounds=120,use_best_model=True);break
            except Exception as e:
                errors.append(dict(device=device,error=str(e)));model=None;log(device+' pilot failed; '+str(e)[:250])
        if model is None:raise RuntimeError('Both CatBoost backends failed')
        catp=np.asarray(model.predict_proba(V));results=[];best=None
        with threadpool_limits(limits=4):
            for weight in [0,.25,.5,.75,1]:
                mix=(1-weight)*p+weight*catp
                for bias in [-.25,0,.25]:
                    pp=adjusted(mix,bias)
                    for cw in [.5,1,2,4]:
                        for lw in [0,.5,1,2]:
                            pred,_,_=decode_scores(pp,c,l,cw,lw);m=metrics(vy,pred);score=(balanced_exact(vy,pred),m['occupied_zone_macro_f1'],-weight)
                            r=dict(tree_weight=weight,bias=bias,count_weight=cw,layout_weight=lw,balanced_two_three_exact=score[0],validation=m);results.append(r)
                            if best is None or score>best[0]:best=(score,r)
        temp=NIGHT/'catboost_pilot_model.partial';joblib.dump(model,temp);temp.replace(NIGHT/'catboost_pilot_model.joblib')
        np.savez_compressed(NIGHT/'catboost_validation_probabilities.npz',catp=catp,mlpp=p,countp=c,layoutp=l,truth=vy)
        report=dict(experiment='CatBoost MultiLogloss diversity pilot',backend=device,parameters=model.get_all_params(),baseline_validation=baseline,best=best[1],top=sorted(results,key=lambda r:(r['balanced_two_three_exact'],r['validation']['occupied_zone_macro_f1'],-r['tree_weight']),reverse=True)[:8],validation_only=True,test_used=False,train_physical_scenes=len(X),elapsed_s=time.time()-started,backend_errors=errors,scope='Exploratory validation result only; no automatic deployment. A fresh evaluation protocol is required before release.')
        atomic('catboost_pilot.json',report);atomic('catboost_state.json',dict(state='completed',pid=os.getpid(),finished=now(),elapsed_s=time.time()-started,best_validation=best[1]));log('CatBoost validation pilot complete; balanced 2/3 exact='+str(best[1]['balanced_two_three_exact']))

if __name__=='__main__':
    try:main()
    except Exception as e:
        atomic('catboost_state.json',dict(state='failed',pid=os.getpid(),updated=now(),error=str(e),traceback=traceback.format_exc()));log('Pilot failure recorded; deployed v2 unchanged.');raise
