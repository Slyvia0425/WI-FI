
import time
import numpy as np
from opt_v2_common import extract,collapse,roughness,decode_scores,MASKS

def infer_v2(b,h):
    start=time.perf_counter()
    base=dict(status='Unknown',regions=[],probabilities=[0.]*9,estimated_region_count=None,confirmed_regions=0,model_version=2)
    try:
        avg,frames=collapse(h);x=extract(h,b['reference'])[None];r=roughness(h,b['reference'])
    except ValueError as e:return dict(base,reason=str(e),elapsed_ms=(time.perf_counter()-start)*1000)
    if r>b['roughness_limit']:return dict(base,reason='signal_roughness_outside_calibration',frames=frames,signal_roughness=r,elapsed_ms=(time.perf_counter()-start)*1000)
    xx=b['scaler'].transform(x);p=np.mean([m.predict_proba(xx) for m in b['models']],axis=0);c=b['count_model'].predict_proba(xx);l=b['layout_model'].predict_proba(xx)
    p=1/(1+np.exp(-(np.log(np.clip(p,1e-7,1-1e-7)/np.clip(1-p,1e-7,1))+b['odds_bias'])));pred,post,_=decode_scores(p,c,l,b['count_weight'],b['layout_weight']);logp=np.log(np.maximum(post,1e-20))/b['temperature'];logp-=logp.max(1,keepdims=True);pp=np.exp(logp);pp/=pp.sum(1,keepdims=True);marg=(pp@__import__('opt_v2_common').MASKS)[0];confidence=float(pp.max());accepted=confidence>=b['layout_gate'] and frames==b['frames']
    regions=[dict(zone=int(z),probability=float(marg[z]),threshold=b['layout_gate'],accepted=accepted) for z in np.flatnonzero(pred[0])]
    alternatives=[dict(zones=np.flatnonzero(MASKS[i]).tolist(),probability=float(pp[0,i])) for i in np.argsort(pp[0])[-3:][::-1]]
    return dict(status='Accepted regions' if accepted else 'Uncertain',reason='structured_signal_layout' if frames==b['frames'] else 'nonstandard_frame_count_not_16_frame_validated',regions=regions,probabilities=marg.tolist(),estimated_region_count=int(pred.sum()),confirmed_regions=len(regions) if accepted else 0,layout_confidence=confidence,count_probabilities=c[0].tolist(),alternative_layouts=alternatives,frames=frames,signal_roughness=r,model_version=2,elapsed_ms=(time.perf_counter()-start)*1000,note='Signal-derived joint center-zone layout. Fixed calibrated simulator only; no individual dimensions or real-world guarantee.')
