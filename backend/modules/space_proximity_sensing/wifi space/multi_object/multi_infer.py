
import time
import numpy as np
from multi_common import NAMES,SIZES,features,center

def predict_probabilities(bundle,X):
    raw=bundle['model'].predict_proba(X);output=[]
    for k,p in enumerate(raw):
        full=np.zeros((len(p),10));classes=bundle['model'].classes_[k];full[:,classes]=p
        z=np.log(np.clip(full,1e-12,1))/bundle.get('temperatures',[1,1,1])[k];z-=z.max(1,keepdims=True);p=np.exp(z);output.append(p/p.sum(1,keepdims=True))
    return np.stack(output,axis=1)

def infer(bundle,h):
    start=time.perf_counter()
    try:X=features(np.asarray(h),bundle['reference'])[None]
    except ValueError as e:return dict(status='Unknown',reason=str(e),objects=[],heads=[],estimated_count=None,confirmed_count=0,elapsed_ms=(time.perf_counter()-start)*1000)
    p=predict_probabilities(bundle,X)[0];distance=float(bundle['neighbors'].kneighbors(bundle['scaler'].transform(X))[0][0,0]);quality=distance<=bundle['distance_limit'];heads=[];objects=[]
    raw=p.argmax(1);conf=p.max(1);duplicates={int(v) for v in raw if v>0 and sum(raw==v)>1}
    for k in range(3):
        label=int(raw[k]);gate=bundle['gates'][k];accepted=bool(quality and conf[k]>=gate and label not in duplicates)
        heads.append(dict(kind=k,name=NAMES[k],raw_label=label,confidence=float(conf[k]),threshold=gate,accepted=accepted,probabilities=p[k].tolist()))
        if label>0 and quality:objects.append(dict(kind=k,name=NAMES[k],zone=label-1,center_m=center(label-1),size_m=SIZES[k].tolist(),confidence=float(conf[k]),accepted=accepted,geometry_note='Canonical size for the predicted size category; actual dimensions are not regressed.'))
    count=int(np.sum(raw>0));fully=quality and count>0 and all(h['accepted'] for h in heads)
    return dict(status='Accepted' if fully else 'Uncertain' if quality else 'Unknown',reason='accepted' if fully else 'unsupported_empty_scene' if quality and count==0 else 'conflicting_object_zones' if duplicates else 'some_size_heads_uncertain' if quality else 'outside_signal_reference_range',objects=objects,heads=heads,estimated_count=count if quality else None,confirmed_count=sum(o['accepted'] for o in objects),raw_count=count,novelty_distance=distance,elapsed_ms=(time.perf_counter()-start)*1000)
