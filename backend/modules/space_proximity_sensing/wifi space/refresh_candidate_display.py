
import json
import numpy as np
import joblib
from coverage_common import DATA,OUT,signal_features,decisions
from coverage_train import noise_seed,evaluate
from learning import observe,classify

def main():
    r=json.loads((OUT/'results.json').read_text(encoding='utf-8'));b=joblib.load(OUT/'model.joblib');rows=json.loads((DATA/'manifest.json').read_text())
    preds=json.loads((OUT/'predictions.json').read_text());byid={p['id']:p for p in preds}
    for split in r['splits']:
        X=[];y=[];xy=[];ids=[]
        for row in [a for a in rows if a['split']==split]:
            with np.load(DATA/'records'/(row['id']+'.npz'),allow_pickle=False) as a:h=a['h']
            h=observe(h,np.random.default_rng(noise_seed(row['id'])));X.append(signal_features(h,b['reference']));y.append(row['zone']);xy.append(row['xy']);ids.append(row['id'])
        met,p,dist,sets,keep=evaluate(b,dict(X=np.array(X),y=np.array(y),xy=np.array(xy)))
        assert met['accepted_count']==r['splits'][split]['accepted_count']
        assert met['accuracy']==r['splits'][split]['accuracy']
        r['splits'][split]=met;r['baseline_comparison'][split]['new']=met
        for i,id_ in enumerate(ids):
            assert byid[id_]['prediction']==(int(p[i].argmax()) if keep[i] else None)
            byid[id_]['candidate_zones']=np.flatnonzero(sets[i]).tolist()
    r['candidate_set_note']='Calibration-derived single-zone evidence gate. If no class meets it, top-two ranked candidates are displayed. Acceptance and all raw/accepted scores are unchanged; candidates have no coverage guarantee.'
    r['candidate_display_refresh']='Display-only fallback checked against every saved evaluation decision; no model, score gate or accepted prediction changed.'
    for name,value in [('results.json',r),('predictions.json',preds)]:
        path=OUT/name;tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
    print('Candidate display refreshed; all accepted predictions unchanged.',flush=True)

if __name__=='__main__':main()
