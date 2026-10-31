
import time
import numpy as np
from learning import zone_center
from coverage_common import signal_features,decisions

def classify_coverage(bundle,h):
    start=time.perf_counter()
    try:x=signal_features(np.asarray(h),bundle['reference'])
    except ValueError as e:return dict(prediction=None,raw_prediction=None,confidence=0,probabilities=[0.0]*9,candidate_zones=[],reason=str(e),center_m=None,elapsed_ms=(time.perf_counter()-start)*1000)
    p,dist,masks,keep=decisions(bundle,x[None]);p=p[0];mask=masks[0];candidates=np.flatnonzero(mask).tolist();index=int(np.argmax(p));accepted=bool(keep[0])
    if dist[0]>bundle['novelty_limit']:reason='outside_signal_reference_range';candidates=[]
    elif len(candidates)>1:reason='ambiguous_zones'
    elif len(candidates)==0:reason='no_supported_zone'
    elif not accepted:reason='low_confidence'
    else:reason='accepted'
    return dict(prediction=index if accepted else None,raw_prediction=index,confidence=float(p.max()),probabilities=p.tolist(),candidate_zones=candidates,reason=reason,novelty_distance=float(dist[0]),center_m=zone_center(index) if accepted else None,elapsed_ms=(time.perf_counter()-start)*1000)
