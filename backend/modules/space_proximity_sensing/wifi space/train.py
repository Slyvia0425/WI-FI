
import json,time,hashlib,warnings,platform
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix,classification_report
from learning import ROOT,CFG,features,observe,zone_center

def metrics(y,p,xy):
    pred=np.argmax(p,axis=1)
    err=np.linalg.norm(np.array([zone_center(int(k)) for k in pred])-xy,axis=1)
    return dict(n=len(y),accuracy=float(accuracy_score(y,pred)),macro_f1=float(f1_score(y,pred,average='macro',labels=list(range(9)),zero_division=0)),confusion_matrix=confusion_matrix(y,pred,labels=list(range(9))).tolist(),per_class=classification_report(y,pred,labels=list(range(9)),output_dict=True,zero_division=0),zone_center_error_m=dict(mean=float(err.mean()),median=float(np.median(err)),p90=float(np.percentile(err,90))),position_floor_m=float(np.linalg.norm(np.array([zone_center(int(k)) for k in y])-xy,axis=1).mean()))

def main():
    data=ROOT/'data';out=ROOT/'outputs';out.mkdir(exist_ok=True)
    manifest=json.loads((data/'manifest.json').read_text());ref=np.load(data/'empty_reference.npy')
    xs={};ys={};positions={};ids={};rawhash={};xysets={}
    for split in dict.fromkeys(r['split'] for r in manifest):
        X=[];y=[];xy=[];names=[];hashes=[]
        for row in [r for r in manifest if r['split']==split]:
            with np.load(data/'records'/(row['id']+'.npz'),allow_pickle=False) as p:h=p['h']
            hashes.append(hashlib.sha256(h.tobytes()).hexdigest())

            seed=int.from_bytes(hashlib.sha256(row['id'].encode()).digest()[:4],'little')
            rng=np.random.default_rng(seed)
            for repeat in range(4 if split=='train' else 1):
                X.append(features(observe(h,rng),ref));y.append(row['zone']);xy.append(row['xy']);names.append(row['id'])
        xs[split]=np.array(X);ys[split]=np.array(y);positions[split]=np.array(xy);ids[split]=names;rawhash[split]=hashes
        xysets[split]={tuple(r['xy']) for r in manifest if r['split']==split}
    overlap={s:len(set(rawhash['train'])&set(rawhash[s])) for s in xs if s!='train'}
    if any(overlap.values()):raise RuntimeError('Duplicate raw channels across split: '+str(overlap))
    if any(xysets['train']&xysets[s] for s in xs if s!='train'):raise RuntimeError('Position leakage')
    candidates={
        'Logistic regression':make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1500,random_state=42)),
        'Random forest':RandomForestClassifier(n_estimators=240,min_samples_leaf=2,max_features=.7,n_jobs=4,random_state=42),
        'MLP (96,48)':make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(96,48),max_iter=400,early_stopping=True,validation_fraction=.15,n_iter_no_change=25,random_state=42,alpha=.05))
    }
    comparison=[];best=None;best_score=-1
    for name,model in candidates.items():
        t=time.perf_counter();model.fit(xs['train'],ys['train']);elapsed=time.perf_counter()-t
        p=model.predict_proba(xs['validation']);score=float(f1_score(ys['validation'],p.argmax(1),average='macro'))
        comparison.append(dict(name=name,validation_macro_f1=score,training_seconds=elapsed));print(comparison[-1],flush=True)
        joblib.dump(model,out/('candidate_'+name.split()[0]+'.joblib'))
        if score>best_score:best_score=score;best=(name,model,p)
    name,model,valp=best


    curve=[]
    for threshold in np.linspace(.6,.95,36):
        mask=valp.max(1)>=threshold
        curve.append(dict(threshold=float(threshold),coverage=float(mask.mean()),accepted=int(mask.sum()),accuracy=float((valp.argmax(1)[mask]==ys['validation'][mask]).mean()) if mask.any() else None))
    eligible=[r for r in curve if r['accepted']>=30 and r['accuracy']>=.9]
    threshold=min(eligible,key=lambda r:r['threshold'])['threshold'] if eligible else .6
    novelty_scaler=StandardScaler().fit(xs['train'])
    novelty_neighbors=NearestNeighbors(n_neighbors=1,n_jobs=1).fit(novelty_scaler.transform(xs['train']))
    val_distance=novelty_neighbors.kneighbors(novelty_scaler.transform(xs['validation']))[0][:,0]
    novelty_limit=float(np.quantile(val_distance,.99))
    bundle=dict(model=model,reference=ref,threshold=threshold,name=name,config=CFG,novelty_scaler=novelty_scaler,novelty_neighbors=novelty_neighbors,novelty_limit=novelty_limit,feature_schema='132 signal-derived features only; no coordinates, labels, IDs, split, or geometry descriptors')
    joblib.dump(bundle,out/'model.joblib.partial');(out/'model.joblib.partial').replace(out/'model.joblib')
    result=dict(scope='Simulation only: one known-size cuboid in one fixed room; zone classification, not semantic furniture imaging',selected_model=name,selection='Maximum validation Macro-F1; no test selection. Original phase-1 tests are now diagnostic, not final claims.',threshold=threshold,novelty_limit=novelty_limit,novelty_rule='Nearest training-observation distance after train-only standardization; limit is 99th percentile of joint-validation distances. Heuristic, not guaranteed OOD detection.',threshold_calibration='Validation coverage vs conditional accuracy, minimum threshold 0.6; confidence is not a calibrated probability of correctness',comparison=comparison,feature_schema=bundle['feature_schema'],feature_dimension=int(xs['train'].shape[1]),train_positions=len(rawhash['train']),train_augmented_observations=len(ys['train']),split_counts={s:len(rawhash[s]) for s in rawhash},duplicate_train_raw_channels=overlap,config=CFG,validation_rejection_curve=curve,splits={},diagnostics={})
    predictions=[]
    for split in xs:
        if split=='train':continue
        p=model.predict_proba(xs[split]);m=metrics(ys[split],p,positions[split]);dist=novelty_neighbors.kneighbors(novelty_scaler.transform(xs[split]))[0][:,0];accept=(p.max(1)>=threshold)&(dist<=novelty_limit)
        m['accepted_count']=int(accept.sum());m['coverage']=float(accept.mean());m['accepted_accuracy']=float((p.argmax(1)[accept]==ys[split][accept]).mean()) if accept.any() else None
        result['diagnostics' if split.startswith('diagnostic') else 'splits'][split]=m
        for i,recordid in enumerate(ids[split]):predictions.append(dict(id=recordid,split=split,true_zone=int(ys[split][i]),xy=positions[split][i].tolist(),probabilities=p[i].tolist(),raw_prediction=int(p[i].argmax()),accepted=bool(accept[i])))

    stress={}
    for snr in [20,10,0]:
        xx=[]
        for row in [r for r in manifest if r['split']=='test']:
            with np.load(data/'records'/(row['id']+'.npz'),allow_pickle=False) as a:h=a['h']
            rng=np.random.default_rng(int.from_bytes(hashlib.sha256((row['id']+str(snr)).encode()).digest()[:4],'little'))
            xx.append(features(observe(h,rng,snr_db=snr,gain_sigma=.35),ref))
        pp=model.predict_proba(xx);stress[str(snr)]=metrics(ys['test'],pp,positions['test']);dd=novelty_neighbors.kneighbors(novelty_scaler.transform(xx))[0][:,0];keep=(pp.max(1)>=threshold)&(dd<=novelty_limit);stress[str(snr)]['coverage']=float(keep.mean());stress[str(snr)]['accepted_accuracy']=float((pp.argmax(1)[keep]==ys['test'][keep]).mean()) if keep.any() else None
    result['noise_stress']=stress
    timings=[]
    from learning import classify
    row=next(r for r in manifest if r['split']=='test')
    with np.load(data/'records'/(row['id']+'.npz'),allow_pickle=False) as a:h=a['h']
    for i in range(55):
        t=time.perf_counter();classify(bundle,h)
        if i>=5:timings.append((time.perf_counter()-t)*1000)
    result['inference_ms']=dict(median=float(np.median(timings)),p95=float(np.percentile(timings,95)),n=50,scope='Warm model, signal features + classification; excludes RT simulation and HTTP')
    result['model_bytes']=(out/'model.joblib').stat().st_size
    result['software']=dict(python=platform.python_version(),numpy=np.__version__)
    (out/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    (out/'predictions.json').write_text(json.dumps(predictions,ensure_ascii=False),encoding='utf-8')
    print('Selected',name,'TEST',result['splits']['test']['macro_f1'],'coverage',result['splits']['test']['coverage'],flush=True)

if __name__=='__main__':main()
