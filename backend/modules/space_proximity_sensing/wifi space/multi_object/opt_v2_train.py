
import copy,json,time,hashlib,warnings
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.exceptions import ConvergenceWarning
from scipy.optimize import minimize_scalar
from opt_v2_common import *
from multi_common import features
from multi_train import seed
from occupancy import proba,met
warnings.filterwarnings('ignore',category=ConvergenceWarning)

def progress(**kw):
    p=OUT2/'training_progress.json';q=p.with_suffix('.partial');q.write_text(json.dumps(kw,indent=2));q.replace(p);print(kw,flush=True)

def feature_set(split,ref):
    cache=OUT2/f'features_{split}.npz'
    if cache.exists():
        with np.load(cache) as a:return {k:a[k] for k in a.files}
    rows=json.loads((DATA2/'manifest.json').read_text());jobs=[(DATA2,r) for r in rows if r['split']==split]
    if split=='train':jobs=[(DATA,r) for r in json.loads((DATA/'manifest.json').read_text()) if r['split']=='train']+jobs
    X=[];Y=[];ids=[];rough=[];single=[];averaged=[]
    for index,(folder,r) in enumerate(jobs):
        path=folder/'records'/(r['id']+'.npz');waiting=False
        while not path.exists():
            if not waiting:progress(state='waiting_for_physical_scenes',split=split,index=index,total=len(jobs));waiting=True
            time.sleep(3)
        with np.load(path,allow_pickle=False) as a:h=a['h']
        rng=np.random.default_rng(seed('optimized-v2-'+r['id']))
        for repeat in range(2 if split=='train' else 1):
            obs=window(h,rng);X.append(extract(obs,ref));Y.append(targets([r])[0]);ids.append(r['id']);rough.append(roughness(obs,ref))
            if split!='train':single.append(features(obs[0],ref));averaged.append(features(obs.mean(0),ref))
        if (index+1)%1000==0:progress(state='extracting',split=split,done=index+1,total=len(jobs))
    d=dict(X=np.array(X),y=np.array(Y),ids=np.array(ids),rough=np.array(rough))
    if split!='train':d.update(single=np.array(single),averaged=np.array(averaged))
    with cache.with_suffix('.partial').open('wb') as f:np.savez_compressed(f,**d)
    cache.with_suffix('.partial').replace(cache);return d

def balanced_exact(y,pred):
    return float(np.mean([np.all(y[y.sum(1)==n]==pred[y.sum(1)==n],axis=1).mean() for n in [2,3]]))

def fit_outer(name,X,y,V,vy,kind,random_state,hidden):
    path=OUT2/(name+'.joblib')
    if path.exists():return joblib.load(path)
    model=MLPClassifier(hidden_layer_sizes=hidden,batch_size=256,alpha=.3,learning_rate_init=.0007,early_stopping=False,random_state=random_state,shuffle=True)
    best=-1;selected=None;best_epoch=0;started=time.time();trace=[]
    classes=np.arange(9 if kind=='occupancy' else 129 if kind=='layout' else 3)
    for epoch in range(1,181):
        model.partial_fit(X,y,classes=classes)
        if epoch%5:continue
        p=model.predict_proba(V)
        if kind=='occupancy':score=float(f1_score(vy,p>=.4,average='macro',zero_division=0))
        elif kind=='layout':score=balanced_exact(vy,MASKS[p.argmax(1)])
        else:score=float(np.mean(p.argmax(1)==vy))
        trace.append(dict(epoch=epoch,score=score));progress(state='training',model=name,epoch=epoch,validation_score=score,best=best,elapsed_s=time.time()-started)
        if score>best+1e-5:best=score;selected=copy.deepcopy(model);best_epoch=epoch
        if epoch>=45 and epoch-best_epoch>=30:break
    joblib.dump(selected,path);(OUT2/(name+'_trace.json')).write_text(json.dumps(trace,indent=2));return selected

def raw_prob(b,X):
    xx=b['scaler'].transform(X);ps=[m.predict_proba(xx) for m in b['models']]
    p=np.mean(ps,axis=0);c=b['count_model'].predict_proba(xx);l=b['layout_model'].predict_proba(xx)
    return p,c,l

def metrics(y,pred):
    m=met(y,pred,.5);m['by_true_count']={str(n):met(y[y.sum(1)==n],pred[y.sum(1)==n],.5) for n in [1,2,3]};return m

def main():
    OUT2.mkdir(exist_ok=True,parents=True);ref=np.load(DATA/'empty_reference.npy');start=time.time()
    tr=feature_set('train',ref);v=feature_set('validation',ref)
    scaler=StandardScaler().fit(tr['X']);X=scaler.transform(tr['X']);V=scaler.transform(v['X']);y=tr['y'];vy=v['y'];selection=OUT2/'selected.joblib'
    with threadpool_limits(limits=4):
        if selection.exists():b=joblib.load(selection)
        else:
            occ=[fit_outer('occupancy_'+str(s),X,y,V,vy,'occupancy',s,(384,192)) for s in [73,91]]
            count=fit_outer('count',X,y.sum(1).astype(int)-1,V,vy.sum(1).astype(int)-1,'count',82,(192,96))
            layout=fit_outer('layout',X,layout_ids(y),V,vy,'layout',83,(512,256))
            b=dict(version=2,name='Calibrated multi-frame ensemble + learned layout decoder',reference=ref,scaler=scaler,models=occ,count_model=count,layout_model=layout,frames=16,feature_dimension=X.shape[1])
            p,c,l=raw_prob(b,v['X']);comparisons=[];best=(-1,-1)

            for bias in [-.5,-.25,0,.25,.5]:
                pp=1/(1+np.exp(-(np.log(np.clip(p,1e-7,1-1e-7)/np.clip(1-p,1e-7,1))+bias)))
                for cw in [0,.5,1,2,4]:
                    for lw in [0,.5,1,2,4]:
                        pred,_,_=decode_scores(pp,c,l,cw,lw);m=met(vy,pred,.5);score=(balanced_exact(vy,pred),m['occupied_zone_macro_f1'])
                        comparisons.append(dict(name=f'count={cw}, layout={lw}, bias={bias}',validation=m,selection_score=score[0]))
                        if score>best:best=score;b.update(count_weight=cw,layout_weight=lw,odds_bias=bias)
            b['comparisons']=sorted(comparisons,key=lambda a:(a['selection_score'],a['validation']['occupied_zone_macro_f1']),reverse=True)[:8]
            joblib.dump(b,selection)
        cal=feature_set('calibration',ref);p,c,l=raw_prob(b,cal['X']);p=1/(1+np.exp(-(np.log(np.clip(p,1e-7,1-1e-7)/np.clip(1-p,1e-7,1))+b['odds_bias'])));pred,post,marg=decode_scores(p,c,l,b['count_weight'],b['layout_weight']);truth_id=layout_ids(cal['y']);logs=np.log(np.maximum(post,1e-20))
        def nll(T):
            zz=logs/T;zz-=zz.max(1,keepdims=True);pp=np.exp(zz);pp/=pp.sum(1,keepdims=True);return -np.log(np.maximum(pp[np.arange(len(pp)),truth_id],1e-20)).mean()
        b['temperature']=float(minimize_scalar(nll,bounds=(.5,4),method='bounded').x)
        calibrated=np.exp((logs-logs.max(1,keepdims=True))/b['temperature']);calibrated/=calibrated.sum(1,keepdims=True);correct=np.all(pred==cal['y'],axis=1);eligible=[]
        for gate in np.linspace(.35,.95,61):
            keep=calibrated.max(1)>=gate
            if keep.sum()>=30 and correct[keep].mean()>=.9:eligible.append(float(gate))
        b['layout_gate']=min(eligible) if eligible else .95;b['roughness_limit']=float(max(np.quantile(cal['rough'],.995)*1.5,.08));b['calibration_note']='Temperature and layout-confidence gate use calibration. Input quality guard is not a real-world OOD detector.'
        joblib.dump(b,OUT2/'model.joblib');progress(state='selected_and_calibrated',decoder={k:b[k] for k in ['count_weight','layout_weight','odds_bias','temperature','layout_gate','roughness_limit']})
        report=dict(version=2,selected_model=b['name'],selection='Independent validation, prioritizing mean complete-layout accuracy for true counts 2 and 3. 1500 new test and 450 boundary scenes sealed until selection. No true count supplied to predictor.',measurement='16 distinct noisy receiver frames of a static synchronized scene; calibrated complex CSI relative to empty-room reference. Single-frame input still accepted, with limited support.',training=dict(physical_scenes=15000,new_scenes=12000,original_training_only=3000,observations=len(X),augmentation=2,features=X.shape[1]),split_counts=dict(train=15000,validation=900,calibration=600,test=1500,boundary_test=450),comparison=b['comparisons'],splits={},baseline_comparison={},model_bytes=(OUT2/'model.joblib').stat().st_size)
        old=joblib.load(OUT/'occupancy.joblib')
        auxiliary=joblib.load(OUT/'model.joblib');aux_report=json.loads((OUT/'results.json').read_text());aux_report['splits']={};aux_report['evaluation_note']='Original experimental size model, reevaluated on the same NEW scenes and 16-frame means as primary v2; not retrained.'
        for split in ['validation','calibration','test','boundary_test']:
            d=v if split=='validation' else cal if split=='calibration' else feature_set(split,ref);p,c,l=raw_prob(b,d['X']);p=1/(1+np.exp(-(np.log(np.clip(p,1e-7,1-1e-7)/np.clip(1-p,1e-7,1))+b['odds_bias'])));pred,post,marg=decode_scores(p,c,l,b['count_weight'],b['layout_weight']);m=metrics(d['y'],pred)
            zz=np.log(np.maximum(post,1e-20))/b['temperature'];zz-=zz.max(1,keepdims=True);pp=np.exp(zz);pp/=pp.sum(1,keepdims=True);keep=pp.max(1)>=b['layout_gate'];quality=d['rough']<=b['roughness_limit'];keep&=quality;m.update(quality_coverage=float(quality.mean()),accepted_layout_coverage=float(keep.mean()),accepted_layout_accuracy=float(np.all(pred[keep]==d['y'][keep],axis=1).mean()) if keep.any() else None);report['splits'][split]=m
            report['baseline_comparison'][split]={}
            for label,key in [('v1_single_frame','single'),('v1_same_16_frame_average','averaged')]:report['baseline_comparison'][split][label]=metrics(d['y'],proba(old['model'],d[key])>=np.array(old['thresholds']))
            if split=='test':
                from multi_infer import predict_probabilities
                from multi_train import metrics as typed_metrics
                rows_by_id={r['id']:r for r in json.loads((DATA2/'manifest.json').read_text())}
                from multi_common import target
                ty=np.array([target(rows_by_id[id]['objects']) for id in d['ids']]);tp=predict_probabilities(auxiliary,d['averaged']);tm=typed_metrics(ty,tp.argmax(2));tm['by_true_count']={str(n):typed_metrics(ty[(ty>0).sum(1)==n],tp.argmax(2)[(ty>0).sum(1)==n]) for n in [1,2,3]};tm['fully_accepted_scene_coverage']=None;aux_report['splits']['test']=tm
            np.savez_compressed(OUT2/f'predictions_{split}.npz',ids=d['ids'],truth=d['y'],pred=pred,posterior=pp,quality=quality,accepted=keep);print(split,m,flush=True)

        times=[]
        from opt_v2_infer import infer_v2
        row=next(r for r in json.loads((DATA2/'manifest.json').read_text()) if r['split']=='test')
        with np.load(DATA2/'records'/(row['id']+'.npz')) as a:obs=window(a['h'],np.random.default_rng(17),30)
        for i in range(35):
            st=time.perf_counter();infer_v2(b,obs)
            if i>=5:times.append((time.perf_counter()-st)*1000)
        report['inference_ms']=dict(median=float(np.median(times)),p95=float(np.percentile(times,95)),scope='Signal features + classifier + confidence; excludes ray tracing, acquisition and HTTP')
        report['selected_decoder']={k:b[k] for k in ['count_weight','layout_weight','odds_bias']}
        (OUT2/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False));progress(state='complete',elapsed_s=time.time()-start,test=report['splits']['test'],boundary=report['splits']['boundary_test'])
        (OUT2/'auxiliary_results.json').write_text(json.dumps(aux_report,indent=2))
if __name__=='__main__':main()
