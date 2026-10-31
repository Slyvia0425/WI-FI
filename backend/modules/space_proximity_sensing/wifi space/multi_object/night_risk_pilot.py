
import os,json,hashlib,copy
from pathlib import Path
import joblib,numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,brier_score_loss
from threadpoolctl import threadpool_limits
from guard_features import make,NAMES
from night_cache import load
from night_common import NIGHT,atomic,stage_lock,log,now
from night_learn import safe_dump
G=NIGHT/'guard_pilot';R=NIGHT/'risk_pilot';R.mkdir(exist_ok=True)

def stats(correct,keep):return dict(accepted_n=int(keep.sum()),coverage=float(keep.mean()),accepted_exact=float(correct[keep].mean()) if keep.any() else None,wrong_accepted=int(np.sum(keep&~correct)))

def main():
    with stage_lock('risk_pilot'),threadpool_limits(limits=4):
        atomic('risk_state.json',dict(state='fitting',pid=os.getpid(),updated=now(),test_used=False))
        b=joblib.load(NIGHT/'model.joblib');models=[joblib.load(G/('zone_'+str(z)+'.joblib')) for z in range(9)]
        va=load('validation');cal=load('calibration');X,r,off=make(b,models,va);C,cr,coff=make(b,models,cal);y=np.all(r['pred']==va['labels'],axis=1);cy=np.all(cr['pred']==cal['labels'],axis=1)
        train,valid=train_test_split(np.arange(len(y)),test_size=400,random_state=2026103013,stratify=y);traces=[];best=None
        candidates=[('Logistic C='+str(c),make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1000,random_state=52))) for c in [.1,1,10]]
        candidates += [('Shallow boosted risk, leaves='+str(leaves),HistGradientBoostingClassifier(max_leaf_nodes=leaves,min_samples_leaf=40,l2_regularization=10,learning_rate=.05,max_iter=100,early_stopping=False,random_state=52)) for leaves in [3,5,7]]
        for name,m in candidates:
            m.fit(X[train],y[train]);p=m.predict_proba(X[valid])[:,1];auc=float(roc_auc_score(y[valid],p));bs=float(brier_score_loss(y[valid],p));traces.append(dict(name=name,development_auc=auc,brier=bs))
            if best is None or (auc,-bs)>best[0]:best=((auc,-bs),name,copy.deepcopy(m))
        model=best[2];model.fit(X,y);p=model.predict_proba(C)[:,1];eligible=[];grid=[]
        for gate in np.linspace(.1,.99,90):
            s=stats(cy,cr['accepted']&(p>=gate));s['gate']=float(gate);grid.append(s)
            if s['accepted_n']>=40 and s['coverage']>=.25 and s['accepted_exact']>=.95:eligible.append(s)
        choice=max(eligible,key=lambda v:v['coverage']) if eligible else None
        baseline_auc=float(roc_auc_score(y[valid],X[valid,0]));passed=bool(choice and best[0][0]>baseline_auc+.005)
        result=dict(state='pilot_complete',completed_at=now(),passed=passed,selected=best[1],feature_names=NAMES,development_split=dict(train=len(train),validation=len(valid),seed=2026103013),development_baseline_confidence_auc=baseline_auc,comparisons=traces,calibration_base=stats(cy,cr['accepted']),calibration_gate=choice,calibration_grid=grid,test_used=False,primary_model_sha256=hashlib.sha256((NIGHT/'model.joblib').read_bytes()).hexdigest(),note='Primary/coordinate checkpoint selection used development validation previously. Error ranker fits that development set; separate600 calibration determines gate. A NEW final test is mandatory before release.')
        safe_dump(dict(model=model,feature_names=NAMES,gate=choice['gate'] if choice else None,coordinate_models=models,primary_model_sha256=result['primary_model_sha256'],spec=result),R/'model.joblib');(R/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');atomic('risk_state.json',dict(state='pilot_complete',pid=os.getpid(),updated=now(),passed=passed,gate=choice,test_used=False));log('Learned reliability pilot completed, passed='+str(passed))

if __name__=='__main__':
    try:main()
    except Exception as e:
        import traceback
        atomic('risk_state.json',dict(state='failed',pid=os.getpid(),updated=now(),error=str(e),traceback=traceback.format_exc()));raise
