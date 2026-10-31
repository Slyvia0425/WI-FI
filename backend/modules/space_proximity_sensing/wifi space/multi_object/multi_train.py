
import json,time,hashlib,joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score
from scipy.optimize import minimize_scalar
from multi_common import ROOT,DATA,OUT,CONFIG,NAMES,noisy,features
from multi_infer import predict_probabilities,infer
from multi_simulate import progress

def seed(id):return int.from_bytes(hashlib.sha256(id.encode()).digest()[:4],'little')

def metrics(y,pred):
    tp=fp=fn=0;true_occ=np.zeros((len(y),9),int);pred_occ=np.zeros_like(true_occ)
    for i,(a,b) in enumerate(zip(y,pred)):
        truth={(k,int(z)) for k,z in enumerate(a) if z};guess={(k,int(z)) for k,z in enumerate(b) if z};tp+=len(truth&guess);fp+=len(guess-truth);fn+=len(truth-guess)
        for z in a:
            if z:true_occ[i,z-1]=1
        for z in b:
            if z:pred_occ[i,z-1]=1
    return dict(n=len(y),typed_object_precision=tp/max(1,tp+fp),typed_object_recall=tp/max(1,tp+fn),typed_object_f1=2*tp/max(1,2*tp+fp+fn),occupied_zone_macro_f1=float(f1_score(true_occ,pred_occ,average='macro',zero_division=0)),count_accuracy=float(np.mean(np.sum(y>0,axis=1)==np.sum(pred>0,axis=1))),exact_typed_layout_accuracy=float(np.mean(np.all(y==pred,axis=1))),per_size_head_accuracy={NAMES[k]:float(np.mean(y[:,k]==pred[:,k])) for k in range(3)},true_objects=int(np.sum(y>0)),predicted_objects=int(np.sum(pred>0)),typed_true_positives=tp)

def main():
    rows=json.loads((DATA/'manifest.json').read_text());ref=np.load(DATA/'empty_reference.npy');sets={};hashsets={}
    progress(state='extracting_features')
    for split in dict.fromkeys(r['split'] for r in rows):
        X=[];Y=[];ids=[];hashes=set()
        for r in [r for r in rows if r['split']==split]:
            with np.load(DATA/'records'/(r['id']+'.npz'),allow_pickle=False) as a:h=a['h']
            hashes.add(hashlib.sha256(h.tobytes()).hexdigest());rng=np.random.default_rng(seed(r['id']))
            for repeat in range(4 if split=='train' else 1):X.append(features(noisy(h,rng),ref));Y.append(r['target']);ids.append(r['id'])
        sets[split]=dict(X=np.array(X),y=np.array(Y),ids=ids,n_positions=len(hashes));hashsets[split]=hashes
    overlap={s:len(hashsets['train']&hs) for s,hs in hashsets.items() if s!='train'}
    if any(overlap.values()):raise RuntimeError('Duplicate raw channels across train/evaluation')
    tr=sets['train'];val=sets['validation'];best=None;score=-1;comparison=[]
    factories=[('Multi-output Extra Trees',lambda:ExtraTreesClassifier(n_estimators=240,min_samples_leaf=2,max_features=.8,n_jobs=4,random_state=42)),('Three MLP heads',lambda:MultiOutputClassifier(make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(192,96),max_iter=280,early_stopping=True,n_iter_no_change=25,random_state=42,alpha=.1)),n_jobs=1))]
    for name,factory in factories:
        progress(state='training',model=name);t=time.perf_counter();m=factory();m.fit(tr['X'],tr['y']);met=metrics(val['y'],m.predict(val['X']));comparison.append(dict(name=name,validation=met,training_s=time.perf_counter()-t));print(comparison[-1],flush=True)
        if met['typed_object_f1']>score:score=met['typed_object_f1'];best=(name,m)
    name,model=best;cal=sets['calibration'];b=dict(model=model,reference=ref,name=name,config=CONFIG,temperatures=[1,1,1],scaler=StandardScaler().fit(tr['X']),gates=[.8,.8,.8])
    cp=predict_probabilities(b,cal['X']);temp_ids=np.arange(len(cp))%2==0;guard_ids=~temp_ids
    for k in range(3):
        logp=np.log(np.clip(cp[temp_ids,k],1e-12,1));yy=cal['y'][temp_ids,k]
        def loss(T):
            z=logp/T;z-=z.max(1,keepdims=True);pp=np.exp(z);pp/=pp.sum(1,keepdims=True);return -np.log(np.maximum(pp[np.arange(len(pp)),yy],1e-12)).mean()
        b['temperatures'][k]=float(minimize_scalar(loss,bounds=(.4,5),method='bounded').x)
    cp=predict_probabilities(b,cal['X']);gates=[]
    for k in range(3):
        p=cp[guard_ids,k];yy=cal['y'][guard_ids,k];eligible=[]
        for gate in np.linspace(.5,.95,46):
            keep=p.max(1)>=gate
            if keep.sum()>=20 and np.mean(p.argmax(1)[keep]==yy[keep])>=.9:eligible.append(float(gate))
        gates.append(min(eligible) if eligible else .9)
    b['gates']=gates;b['neighbors']=NearestNeighbors(n_neighbors=1,n_jobs=1).fit(b['scaler'].transform(tr['X'][::4]));dist=b['neighbors'].kneighbors(b['scaler'].transform(cal['X'][guard_ids]))[0][:,0];b['distance_limit']=float(np.quantile(dist,.99))
    joblib.dump(b,OUT/'model.joblib.partial');(OUT/'model.joblib.partial').replace(OUT/'model.joblib')
    result=dict(scope='Simulated 1–3 cuboids, distinct size bands and center zones. Category-level dimensions, not arbitrary furniture reconstruction.',selected_model=name,comparison=comparison,config=CONFIG,feature_dimension=544,train_scenes=3000,train_augmented_observations=len(tr['y']),split_counts={s:d['n_positions'] for s,d in sets.items()},duplicate_channels=overlap,gates=gates,temperatures=b['temperatures'],distance_limit=b['distance_limit'],selection='Validation typed-object F1 only; independent calibration split for score temperature and gates; test not used for tuning.',splits={})
    predictions=[]
    for split,d in sets.items():
        if split=='train':continue
        p=predict_probabilities(b,d['X']);raw=p.argmax(2);met=metrics(d['y'],raw);dd=b['neighbors'].kneighbors(b['scaler'].transform(d['X']))[0][:,0];head_keep=(p.max(2)>=np.array(gates))&(dd[:,None]<=b['distance_limit']);fully=head_keep.all(1)
        for i,a in enumerate(raw):
            if len(set(a[a>0]))!=sum(a>0):fully[i]=False
        fully[np.sum(raw>0,axis=1)==0]=False
        met['fully_accepted_scene_coverage']=float(fully.mean());met['fully_accepted_exact_accuracy']=float(np.all(raw[fully]==d['y'][fully],axis=1).mean()) if fully.any() else None
        met['by_true_count']={str(n):metrics(d['y'][np.sum(d['y']>0,axis=1)==n],raw[np.sum(d['y']>0,axis=1)==n]) for n in [1,2,3]}
        result['splits'][split]=met
        for i,id in enumerate(d['ids']):predictions.append(dict(id=id,split=split,true=d['y'][i].tolist(),raw=raw[i].tolist(),probabilities=p[i].tolist(),fully_accepted=bool(fully[i])))
        print(split,met,flush=True)
    times=[]
    row=next(r for r in rows if r['split']=='test')
    with np.load(DATA/'records'/(row['id']+'.npz'),allow_pickle=False) as a:h=a['h']
    for i in range(35):
        t=time.perf_counter();infer(b,h)
        if i>=5:times.append((time.perf_counter()-t)*1000)
    result['inference_ms']=dict(median=float(np.median(times)),p95=float(np.percentile(times,95)),scope='Features, classifier and rejection; excludes joint RT and HTTP');result['model_bytes']=(OUT/'model.joblib').stat().st_size
    for filename,value in [('results.json',result),('predictions.json',predictions)]:
        p=OUT/filename;tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
    progress(state='trained',model=name,test_f1=result['splits']['test']['typed_object_f1']);print('Multi-object model ready',flush=True)

if __name__=='__main__':main()
