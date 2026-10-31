
import json,time,joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,precision_score,recall_score
from multi_common import DATA,OUT,features,noisy
from multi_train import seed

def occupied_target(heads):
    y=np.zeros((len(heads),9),dtype=int)
    for i,row in enumerate(heads):
        for label in row:
            if label:y[i,label-1]=1
    return y

def proba(model,X):
    p=model.predict_proba(X)
    if isinstance(p,list):return np.stack([a[:,list(model.classes_[k]).index(1)] for k,a in enumerate(p)],axis=1)
    return np.asarray(p)

def met(y,p,thresholds):
    pred=p>=thresholds;count=y.sum(1)
    return dict(n=len(y),occupied_zone_macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),occupied_zone_micro_f1=float(f1_score(y,pred,average='micro',zero_division=0)),zone_precision=float(precision_score(y,pred,average='micro',zero_division=0)),zone_recall=float(recall_score(y,pred,average='micro',zero_division=0)),count_accuracy=float(np.mean(pred.sum(1)==count)),exact_occupied_layout_accuracy=float(np.all(y==pred,axis=1).mean()),per_zone_f1=f1_score(y,pred,average=None,zero_division=0).tolist())

def infer_occupancy(bundle,h):
    if bundle.get('version')==5:
        from home_v5_infer import infer_v5
        return infer_v5(bundle,h)
    if bundle.get('version')==4:
        from shared_v4_infer import infer_v4
        return infer_v4(bundle,h)
    if bundle.get('version')==3:
        from night_infer import infer_v3
        return infer_v3(bundle,h)
    if bundle.get('version')==2:
        from opt_v2_infer import infer_v2
        return infer_v2(bundle,h)
    start=time.perf_counter()
    try:X=features(np.asarray(h),bundle['reference'])[None]
    except ValueError as e:return dict(status='Unknown',reason=str(e),regions=[],probabilities=[0.0]*9,estimated_region_count=None,confirmed_regions=0,elapsed_ms=(time.perf_counter()-start)*1000)
    p=proba(bundle['model'],X)[0];distance=float(bundle['neighbors'].kneighbors(bundle['scaler'].transform(X))[0][0,0]);quality=distance<=bundle['distance_limit'];raw=p>=np.array(bundle['thresholds']);regions=[]
    if quality:
        for z in np.flatnonzero(raw):regions.append(dict(zone=int(z),probability=float(p[z]),threshold=bundle['thresholds'][z],accepted=bool(p[z]>=bundle['positive_gates'][z])))
    full=quality and 1<=len(regions)<=3 and all(r['accepted'] for r in regions)
    return dict(status='Accepted regions' if full else 'Uncertain' if quality else 'Unknown',reason='region_proposals' if quality else 'outside_signal_reference_range',regions=regions,probabilities=p.tolist(),estimated_region_count=int(raw.sum()) if quality else None,confirmed_regions=sum(r['accepted'] for r in regions),novelty_distance=distance,elapsed_ms=(time.perf_counter()-start)*1000,note='Occupied center zones only. Does not assign object dimensions or certify that no other object is present.')

def train():
    rows=json.loads((DATA/'manifest.json').read_text());base=joblib.load(OUT/'model.joblib');ds={};ref=base['reference']
    for split in ['train','validation','calibration','test']:
        X=[];Y=[]
        for r in [r for r in rows if r['split']==split]:
            with np.load(DATA/'records'/(r['id']+'.npz'),allow_pickle=False) as a:h=a['h']
            rng=np.random.default_rng(seed(r['id']))
            for i in range(4 if split=='train' else 1):X.append(features(noisy(h,rng),ref));Y.append(r['target'])
        ds[split]=dict(X=np.array(X),y=occupied_target(Y),count=np.sum(np.array(Y)>0,axis=1))
    comparison=[];best=None;best_score=-1
    for name,model in [('Occupancy Extra Trees',ExtraTreesClassifier(n_estimators=260,min_samples_leaf=2,max_features=.8,n_jobs=4,random_state=42)),('Occupancy MLP',make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(256,128),max_iter=320,early_stopping=True,n_iter_no_change=30,random_state=42,alpha=.1)))]:
        start=time.perf_counter();model.fit(ds['train']['X'],ds['train']['y']);p=proba(model,ds['validation']['X']);th=[]
        for z in range(9):th.append(max(np.linspace(.2,.75,56),key=lambda v:f1_score(ds['validation']['y'][:,z],p[:,z]>=v,zero_division=0)))
        score=met(ds['validation']['y'],p,th);comparison.append(dict(name=name,validation=score,training_s=time.perf_counter()-start));print(comparison[-1],flush=True)
        if score['occupied_zone_macro_f1']>best_score:best_score=score['occupied_zone_macro_f1'];best=(name,model,th)
    name,model,th=best;calp=proba(model,ds['calibration']['X']);gates=[]
    for z in range(9):
        eligible=[]
        for g in np.linspace(th[z],.95,45):
            mask=calp[:,z]>=g
            if mask.sum()>=10 and ds['calibration']['y'][mask,z].mean()>=.9:eligible.append(float(g))
        gates.append(min(eligible) if eligible else .9)
    b=dict(model=model,reference=ref,name=name,thresholds=list(map(float,th)),positive_gates=gates,neighbors=base['neighbors'],scaler=base['scaler'],distance_limit=base['distance_limit'])
    joblib.dump(b,OUT/'occupancy.joblib.partial');(OUT/'occupancy.joblib.partial').replace(OUT/'occupancy.joblib')
    r=dict(task='Multi-label occupied center zones; no per-object size assignment',selected_model=name,selection='Validation Macro-F1 with validation-selected per-zone thresholds. Positive evidence gates use calibration; test untouched for selection.',comparison=comparison,thresholds=b['thresholds'],positive_gates=gates,splits={})
    for split in ['validation','calibration','test']:
        d=ds[split];p=proba(model,d['X']);m=met(d['y'],p,th);m['by_true_count']={str(n):met(d['y'][d['count']==n],p[d['count']==n],th) for n in [1,2,3]};r['splits'][split]=m;print(split,m,flush=True)
    r['model_bytes']=(OUT/'occupancy.joblib').stat().st_size
    (OUT/'occupancy_results.json').write_text(json.dumps(r,indent=2));print('Occupancy model ready',flush=True)

if __name__=='__main__':train()
