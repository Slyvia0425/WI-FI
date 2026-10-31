
import json,time,joblib
import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from opt_v2_common import *
from multi_train import seed
from occupancy import met,proba

def main():
    OUT2.mkdir(exist_ok=True,parents=True);ref=np.load(DATA/'empty_reference.npy');rows=json.loads((DATA/'manifest.json').read_text());sets={}
    for split in ['train','validation']:
        rr=[r for r in rows if r['split']==split];X=[];Y=[]
        for r in rr:
            with np.load(DATA/'records'/(r['id']+'.npz')) as a:h=a['h']
            rng=np.random.default_rng(seed(r['id']))
            for repeat in range(2 if split=='train' else 1):X.append(extract(window(h,rng),ref));Y.append(targets([r])[0])
        sets[split]=(np.array(X),np.array(Y));print('features',split,len(X),flush=True)
    results=[]
    with threadpool_limits(limits=4):
        for name,m in [('Phase-calibrated MLP',make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(384,192),max_iter=180,early_stopping=True,n_iter_no_change=18,random_state=73,alpha=.3)))]:
            start=time.time();m.fit(*sets['train']);p=proba(m,sets['validation'][0]);y=sets['validation'][1];th=[]
            from sklearn.metrics import f1_score
            for z in range(9):th.append(max(np.linspace(.2,.75,56),key=lambda v:f1_score(y[:,z],p[:,z]>=v,zero_division=0)))
            decoded,_,_=decode_scores(p,np.ones((len(p),3))/3,count_weight=0)
            r=dict(name=name,validation=met(y,p,th),structured_validation=met(y,decoded,.5),training_s=time.time()-start);results.append(r);print(r,flush=True);joblib.dump(dict(model=m,reference=ref,thresholds=th),OUT2/('pilot_'+name.replace(' ','_')+'.joblib'))
    (OUT2/'pilot_calibrated.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
