
import json,time,hashlib,platform
import numpy as np
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score
from scipy.optimize import minimize_scalar
from learning import CFG,observe,classify,features
from coverage_common import ROOT,DATA,OUT,signal_features,probabilities,decisions
from train import metrics
from coverage_generate import progress

def noise_seed(recordid,suffix=''):
    return int.from_bytes(hashlib.sha256((recordid+suffix).encode()).digest()[:4],'little')

def load_data():
    rows=json.loads((DATA/'manifest.json').read_text());ref=np.load(DATA/'empty_reference.npy');sets={};hashes={};coords={}
    for split in dict.fromkeys(r['split'] for r in rows):
        records=[r for r in rows if r['split']==split];X=[];A=[];y=[];xy=[];ids=[]
        hashes[split]=set();coords[split]=set()
        for r in records:
            with np.load(DATA/'records'/(r['id']+'.npz'),allow_pickle=False) as a:h=a['h']
            hashes[split].add(hashlib.sha256(h.tobytes()).hexdigest());coords[split].add(tuple(r['xy']))
            rng=np.random.default_rng(noise_seed(r['id']))
            for repeat in range(4 if split=='train' else 1):
                hh=observe(h,rng);X.append(signal_features(hh,ref));A.append(features(hh,ref));y.append(r['zone']);xy.append(r['xy']);ids.append(r['id'])
        sets[split]=dict(X=np.array(X),A=np.array(A),y=np.array(y),xy=np.array(xy),ids=ids,n_positions=len(records))
    overlap={s:len(hashes['train']&hashes[s]) for s in sets if s!='train'}
    if any(overlap.values()) or any(coords['train']&coords[s] for s in sets if s!='train'):raise RuntimeError('Split leakage')
    return rows,ref,sets,overlap

def evaluate(bundle,d):
    p,dist,masks,keep=decisions(bundle,d['X']);m=metrics(d['y'],p,d['xy']);pred=p.argmax(1)
    m.update(coverage=float(keep.mean()),accepted_count=int(keep.sum()),accepted_accuracy=float((pred[keep]==d['y'][keep]).mean()) if keep.any() else None,candidate_set_coverage=float(masks[np.arange(len(masks)),d['y']].mean()),mean_candidate_count=float(masks.sum(1).mean()),ambiguous_fraction=float((masks.sum(1)>1).mean()),novelty_rejected_fraction=float((dist>bundle['novelty_limit']).mean()))
    return m,p,dist,masks,keep

def main():
    OUT.mkdir(parents=True,exist_ok=True);progress(state='loading_features')
    rows,ref,ds,overlap=load_data();tr=ds['train'];val=ds['validation']
    cache={s:{k:v for k,v in d.items() if k in ['X','A','y','xy']} for s,d in ds.items()}

    np.savez_compressed(OUT/'validation_features.npz',X=val['X'],y=val['y'])
    factories=[('Logistic regression',lambda:make_pipeline(StandardScaler(),LogisticRegression(max_iter=1500,C=1,random_state=42))),('Random forest',lambda:RandomForestClassifier(n_estimators=240,min_samples_leaf=3,max_features=.7,n_jobs=4,random_state=42)),('Extra trees',lambda:ExtraTreesClassifier(n_estimators=280,min_samples_leaf=3,max_features=.85,n_jobs=4,random_state=42)),('MLP (192,96)',lambda:make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(192,96),max_iter=350,early_stopping=True,n_iter_no_change=30,validation_fraction=.12,random_state=42,alpha=.1)))]
    comparison=[];best_score=-1;best_model=None;best_name=None
    for name,factory in factories:
        progress(state='training',model=name);start=time.perf_counter();m=factory();m.fit(tr['X'],tr['y']);p=m.predict_proba(val['X']);score=float(f1_score(val['y'],p.argmax(1),average='macro'))
        comparison.append(dict(name=name,validation_macro_f1=score,training_seconds=time.perf_counter()-start));print(comparison[-1],flush=True)
        if score>best_score:best_score=score;best_model=m;best_name=name
    progress(state='calibrating',model=best_name)
    temp=ds['cal_temperature'];guard=ds['cal_guard'];logp=np.log(np.clip(best_model.predict_proba(temp['X']),1e-12,1))
    def objective(T):
        z=logp/T;z-=z.max(1,keepdims=True);p=np.exp(z);p/=p.sum(1,keepdims=True)
        return -float(np.log(np.maximum(p[np.arange(len(p)),temp['y']],1e-12)).mean())
    temperature=float(minimize_scalar(objective,bounds=(.35,6),method='bounded').x)
    b=dict(protocol=3,model=best_model,reference=ref,temperature=temperature,name=best_name,config=CFG,feature_version='amplitude_frequency_correlation_204',threshold=.6,novelty_scaler=StandardScaler().fit(tr['X']))
    b['novelty_neighbors']=NearestNeighbors(n_neighbors=1,n_jobs=1).fit(b['novelty_scaler'].transform(tr['X'][::4]))
    gd=b['novelty_neighbors'].kneighbors(b['novelty_scaler'].transform(guard['X']))[0][:,0];b['novelty_limit']=float(np.quantile(gd,.99))
    gp=probabilities(b,guard['X']);true_p=gp[np.arange(len(gp)),guard['y']]
    rank=min(len(true_p),int(np.ceil((len(true_p)+1)*.90)))
    b['set_cutoff']=float(max(1e-9,1-np.sort(1-true_p)[rank-1]))
    masks=gp>=b['set_cutoff'];curve=[]
    for threshold in np.linspace(.5,.95,46):
        keep=(gp.max(1)>=threshold)&(gd<=b['novelty_limit'])&(masks.sum(1)==1)
        curve.append(dict(threshold=float(threshold),coverage=float(keep.mean()),accepted=int(keep.sum()),accuracy=float((gp.argmax(1)[keep]==guard['y'][keep]).mean()) if keep.any() else None))
    eligible=[a for a in curve if a['accepted']>=30 and a['accuracy']>=.9]
    b['threshold']=min(eligible,key=lambda a:a['threshold'])['threshold'] if eligible else .9
    baseline=joblib.load(ROOT/'outputs/model.joblib');oldval=baseline['model'].predict_proba(val['A']);old_val_f1=float(f1_score(val['y'],oldval.argmax(1),average='macro'))
    model_path=OUT/'model.joblib';joblib.dump(b,OUT/'model.joblib.partial');(OUT/'model.joblib.partial').replace(model_path)
    result=dict(protocol=3,scope='Full physically valid placement support for one known-size object; simulated channels only',selection='Validation only. Temperature and acceptance checks use separate calibration sets; frozen tests are not used to tune parameters.',selected_model=best_name,threshold=b['threshold'],threshold_target_met=bool(eligible),temperature=temperature,set_cutoff=b['set_cutoff'],novelty_limit=b['novelty_limit'],candidate_set_note='Calibration-derived single-zone evidence gate. If no class meets the gate, show top-two ranked candidates without accepting a location. Neither candidates nor shifted-domain confidence have a coverage guarantee.',comparison=comparison,feature_dimension=204,feature_schema='132 amplitude features plus 72 phase-offset-invariant within-link frequency correlations. No coordinates, geometry descriptors, IDs or target labels.',train_positions=tr['n_positions'],train_augmented_observations=len(tr['y']),split_counts={s:d['n_positions'] for s,d in ds.items()},duplicate_train_raw_channels=overlap,config=CFG,validation_rejection_curve=curve,splits={},noise_stress={},baseline_comparison={},validation_gain=best_score-old_val_f1,old_validation_macro_f1=old_val_f1,activation_rule='Validation Macro-F1 improvement >= 0.05 over the previous model on identical new validation signals.',activation_recommended=best_score-old_val_f1>=.05)
    predictions=[]
    for split,d in ds.items():
        if split=='train':continue
        met,p,dist,sets,keep=evaluate(b,d);result['splits'][split]=met
        oldp=baseline['model'].predict_proba(d['A']);oldmetrics=metrics(d['y'],oldp,d['xy'])
        olddist=baseline['novelty_neighbors'].kneighbors(baseline['novelty_scaler'].transform(d['A']))[0][:,0]
        oldkeep=(oldp.max(1)>=baseline['threshold'])&(olddist<=baseline['novelty_limit'])
        oldmetrics.update(coverage=float(oldkeep.mean()),accepted_accuracy=float((oldp.argmax(1)[oldkeep]==d['y'][oldkeep]).mean()) if oldkeep.any() else None)
        result['baseline_comparison'][split]=dict(old=oldmetrics,new=met)
        for i,id_ in enumerate(d['ids']):predictions.append(dict(id=id_,split=split,zone=int(d['y'][i]),xy=d['xy'][i].tolist(),prediction=int(p[i].argmax()) if keep[i] else None,raw_prediction=int(p[i].argmax()),candidate_zones=np.flatnonzero(sets[i]).tolist(),confidence=float(p[i].max()),novelty_distance=float(dist[i])))
        print(split,'accuracy',met['accuracy'],'coverage',met['coverage'],'accepted',met['accepted_accuracy'],flush=True)
    for snr in [20,10,0]:
        x=[]
        for r in [r for r in rows if r['split']=='test']:
            with np.load(DATA/'records'/(r['id']+'.npz'),allow_pickle=False) as a:h=a['h']
            hh=observe(h,np.random.default_rng(noise_seed(r['id'],str(snr))),snr_db=snr);x.append(signal_features(hh,ref))
        d=dict(ds['test']);d['X']=np.array(x);result['noise_stress'][str(snr)]=evaluate(b,d)[0]

    from simulate import build_scene,response
    scene,obj,solver=build_scene();obj.position=[2,3,.6];hh=observe(response(scene,solver,17),np.random.default_rng(42),snr_db=30)
    from coverage_infer import classify_coverage
    result['reported_boundary_example']=dict(xy=[2,3],true_zone=4,old=classify(baseline,hh),new=classify_coverage(b,hh),purpose='Diagnostic only, not training or a headline score.')
    row=next(r for r in rows if r['split']=='test')
    with np.load(DATA/'records'/(row['id']+'.npz'),allow_pickle=False) as a:h=a['h']
    times=[]
    for i in range(55):
        start=time.perf_counter();classify_coverage(b,h)
        if i>=5:times.append((time.perf_counter()-start)*1000)
    result['inference_ms']=dict(median=float(np.median(times)),p95=float(np.percentile(times,95)),n=50,scope='Warm feature extraction, classifier, temperature and rejection; excludes RT and HTTP')
    result['model_bytes']=model_path.stat().st_size
    if hasattr(best_model,'steps') and hasattr(best_model.steps[-1][1],'coefs_'):
        net=best_model.steps[-1][1];result['learned_parameters']=sum(a.size for a in net.coefs_)+sum(a.size for a in net.intercepts_)
    result['software']=dict(python=platform.python_version(),numpy=np.__version__)
    for filename,value in [('results.json',result),('predictions.json',predictions)]:
        pth=OUT/filename;tmp=pth.with_suffix('.partial');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(pth)
    progress(state='trained',model=best_name,validation_gain=result['validation_gain'],activation_recommended=result['activation_recommended'])
    print('Finished; activation recommended:',result['activation_recommended'],flush=True)

if __name__=='__main__':main()
