
import os,json,time,copy,hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import joblib,numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from night_common import NIGHT,atomic,now,log,stage_lock
from night_cache import load,jobs_for
from night_learn import safe_dump
from night_infer import batch
G=NIGHT/'guard_pilot';G.mkdir(exist_ok=True)
SPEC=dict(version=1,view='phase_free',hidden=[128,64],alpha=1.,learning_rate=.001,batch_size=128,max_epochs=160,patience=25,minimum_epochs=45,evaluate_every=5,seed=2026103012,goal='Estimate within-zone center offsets from CSI to assess internal-boundary ambiguity; no change to classifier or use of its old final test.')

def positions(split):
    jobs=jobs_for(split);a=np.full((len(jobs),9,2),np.nan,np.float32)
    for i,(_,r) in enumerate(jobs):
        for o in r['objects']:
            z=o['zone'];a[i,z]=[o['x']-(1+2*(z%3)),o['y']-(1+2*(z//3))]
    return a

def fit(z):
    tr=load('train','phase_free');va=load('validation','phase_free');X=tr['phase_free'][::2];V=va['phase_free'];y=positions('train')[:,z];vy=positions('validation')[:,z];keep=np.isfinite(y).all(1);vk=np.isfinite(vy).all(1);y=y[keep];vy=vy[vk]
    path=G/('zone_'+str(z)+'.joblib');source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();fingerprint=hashlib.sha256(json.dumps(dict(spec=SPEC,z=z,train=tr['meta']['fingerprint'],validation=va['meta']['fingerprint'],source=source),sort_keys=True).encode()).hexdigest()
    if path.exists():
        b=joblib.load(path);assert b['fingerprint']==fingerprint;return b['report']
    start=time.time();scaler=StandardScaler().fit(X[keep]);X=scaler.transform(X[keep]);V=scaler.transform(V[vk]);m=MLPRegressor(hidden_layer_sizes=tuple(SPEC['hidden']),alpha=SPEC['alpha'],learning_rate_init=SPEC['learning_rate'],batch_size=SPEC['batch_size'],random_state=SPEC['seed']+z,early_stopping=False);best=None;score=1e9;best_epoch=0;trace=[]
    with threadpool_limits(limits=2):
        for epoch in range(1,SPEC['max_epochs']+1):
            m.partial_fit(X,y)
            if epoch%SPEC['evaluate_every']:continue
            pred=np.clip(m.predict(V),-1,1);mae=float(np.abs(pred-vy).mean());trace.append(dict(epoch=epoch,mae_m=mae))
            if mae<score-1e-5:score=mae;best=copy.deepcopy(m);best_epoch=epoch
            if epoch>=SPEC['minimum_epochs'] and epoch-best_epoch>=SPEC['patience']:break
    if hasattr(best,'_optimizer'):del best._optimizer
    report=dict(zone=z,n_train=len(y),n_validation=len(vy),validation_coordinate_mae_m=score,zone_center_baseline_mae_m=float(np.abs(vy).mean()),best_epoch=best_epoch,elapsed_s=time.time()-start,trace=trace)
    safe_dump(dict(model=best,scaler=scaler,zone=z,fingerprint=fingerprint,report=report),path);log('Boundary pilot zone'+str(z+1)+' validation MAE='+str(score));return report

def risk(models,X,pred):
    scores=np.full(len(X),np.inf);estimated=np.full((len(X),9,2),np.nan)
    for z,m in enumerate(models):
        ix=np.flatnonzero(pred[:,z])
        if not len(ix):continue
        offset=np.clip(m['model'].predict(m['scaler'].transform(X[ix])),-1,1);estimated[ix,z]=offset
        col=z%3;row=z//3
        dx=1-offset[:,0] if col==0 else 1+offset[:,0] if col==2 else 1-np.abs(offset[:,0])
        dy=1-offset[:,1] if row==0 else 1+offset[:,1] if row==2 else 1-np.abs(offset[:,1])
        scores[ix]=np.minimum(scores[ix],np.minimum(dx,dy))
    return np.nan_to_num(scores,nan=0,posinf=0),estimated

def summarize(y,r,scores,margin):
    correct=np.all(y==r['pred'],axis=1);keep=r['accepted']&(scores>=margin)
    return dict(n=len(y),accepted_n=int(keep.sum()),coverage=float(keep.mean()),accepted_exact=float(correct[keep].mean()) if keep.any() else None,wrong_accepted=int(np.sum(keep&~correct)),raw_exact=float(correct.mean()))

def main():
    with stage_lock('guard_pilot'):
        atomic('guard_state.json',dict(state='training',pid=os.getpid(),updated=now(),spec=SPEC,test_used=False))
        with ProcessPoolExecutor(max_workers=3) as pool:reports=list(pool.map(fit,range(9)))
        models=[joblib.load(G/('zone_'+str(z)+'.joblib')) for z in range(9)];b=joblib.load(NIGHT/'model.joblib');results={};cached={}
        with threadpool_limits(limits=4):
            for split in ['validation','calibration']:
                d=load(split);r=batch(b,d);scores,est=risk(models,d['phase_free'],r['pred']);correct=np.all(d['labels']==r['pred'],axis=1);base=summarize(d['labels'],r,scores,0);cached[split]=(d,r,scores)
                results[split]=dict(base=base,risk_auc=float(roc_auc_score(correct,scores)) if len(np.unique(correct))==2 else None,margin_grid=[dict(margin=float(m),**summarize(d['labels'],r,scores,float(m))) for m in np.linspace(0,.8,81)])
        eligible=[v for v in results['calibration']['margin_grid'] if v['accepted_n']>=40 and v['coverage']>=.25 and v['accepted_exact']>=.95]
        margin=min((r['margin'] for r in eligible),default=None)
        improves=np.mean([r['validation_coordinate_mae_m'] for r in reports])<np.mean([r['zone_center_baseline_mae_m'] for r in reports])
        valid=None
        if margin is not None:
            d,r,scores=cached['validation'];valid=summarize(d['labels'],r,scores,margin);improves&=valid['coverage']>=.25 and valid['accepted_exact']>results['validation']['base']['accepted_exact']
        passed=bool(eligible and improves)
        result=dict(state='pilot_complete',completed_at=now(),passed=passed,selected_margin_m=margin,spec=SPEC,regressors=reports,validation_at_selected_margin=valid,results=results,primary_model_sha256=hashlib.sha256((NIGHT/'model.joblib').read_bytes()).hexdigest(),test_used=False,next_step='If passed, freeze guard, generate NEW final geometries and compare abstention on those. Do not tune against v3 final test or user examples.')
        (G/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');atomic('guard_state.json',dict(state='pilot_complete',pid=os.getpid(),updated=now(),passed=passed,margin=margin,test_used=False));log('Boundary guard pilot complete; independent new final test required before deployment. Passed='+str(passed))

if __name__=='__main__':
    try:main()
    except Exception as e:
        import traceback
        atomic('guard_state.json',dict(state='failed',pid=os.getpid(),updated=now(),error=str(e),traceback=traceback.format_exc()));raise
