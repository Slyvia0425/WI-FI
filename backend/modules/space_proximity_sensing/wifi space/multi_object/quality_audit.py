
import json,joblib
import numpy as np
from multi_common import DATA,OUT,features,noisy
from multi_train import seed
from occupancy import proba,occupied_target

def main():
    rows=[r for r in json.loads((DATA/'manifest.json').read_text()) if r['split']=='test'];b=joblib.load(OUT/'occupancy.joblib');X=[];Y=[]
    for r in rows:
        with np.load(DATA/'records'/(r['id']+'.npz'),allow_pickle=False) as a:h=a['h']
        X.append(features(noisy(h,np.random.default_rng(seed(r['id']))),b['reference']));Y.append(r['target'])
    X=np.array(X);y=occupied_target(Y);p=proba(b['model'],X);distance=b['neighbors'].kneighbors(b['scaler'].transform(X))[0][:,0];valid=distance<=b['distance_limit'];strong=(p>=np.maximum(b['thresholds'],b['positive_gates']))&valid[:,None]
    d=dict(n=len(rows),quality_accepted_scene_fraction=float(valid.mean()),strong_region_count=int(strong.sum()),strong_region_precision=float(y[strong].mean()) if strong.any() else None,strong_region_recall=float(np.sum(strong*y)/y.sum()),note='Strong region precision does not certify complete layouts. Raw benchmark metrics remain unchanged.')
    (OUT/'quality_audit.json').write_text(json.dumps(d,indent=2));print(d,flush=True)

if __name__=='__main__':main()
