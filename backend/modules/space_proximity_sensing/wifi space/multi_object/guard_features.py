
import numpy as np
from night_infer import batch,expert_prob,posterior
from opt_v2_common import MASKS

NAMES=['layout_confidence','layout_gap','layout_entropy','free_confidence','free_gap','free_entropy','expert_hamming','posterior_js','expert_same_layout','count_confidence','free_count_confidence','count_disagreement','internal_boundary_min_m','internal_boundary_mean_m','offset_overshoot_m','offset_abs_mean','free_roughness','temporal_coherence']

def make(b,models,d):
    r=batch(b,d);pp=r['posterior'];free_pred,fp,fc=expert_prob(b,'phase_free',d['phase_free']);fp=posterior(fp,b['experts']['phase_free']['temperature']);sp=np.sort(pp,axis=1);sf=np.sort(fp,axis=1);mid=(pp+fp)/2
    def entropy(p):return -(p*np.log(np.maximum(p,1e-20))).sum(1)
    js=entropy(mid)-.5*(entropy(pp)+entropy(fp));n=len(pp);dist=np.full((n,9),np.nan);offsets=np.full((n,9,2),np.nan)
    for z,m in enumerate(models):
        ix=np.flatnonzero(r['pred'][:,z])
        if not len(ix):continue
        off=m['model'].predict(m['scaler'].transform(d['phase_free'][ix]));offsets[ix,z]=off
        x=np.clip(off[:,0],-1,1);y=np.clip(off[:,1],-1,1);col=z%3;row=z//3
        dx=1-x if col==0 else 1+x if col==2 else 1-np.abs(x);dy=1-y if row==0 else 1+y if row==2 else 1-np.abs(y);dist[ix,z]=np.minimum(dx,dy)
    abs_off=np.abs(offsets);overshoot=np.nanmax(np.maximum(abs_off-1,0),axis=(1,2));features=np.column_stack([sp[:,-1],sp[:,-1]-sp[:,-2],entropy(pp),sf[:,-1],sf[:,-1]-sf[:,-2],entropy(fp),np.sum(r['pred']!=free_pred,axis=1),js,np.all(r['pred']==free_pred,axis=1),r['count_probabilities'].max(1),fc.max(1),r['count_probabilities'].argmax(1)!=fc.argmax(1),np.nanmin(dist,axis=1),np.nanmean(dist,axis=1),overshoot,np.nanmean(abs_off,axis=(1,2)),d['quality'][:,3],d['quality'][:,1]])
    if features.shape!=(len(pp),len(NAMES)) or not np.isfinite(features).all():raise ValueError('Invalid reliability features')
    return features.astype(np.float32),r,offsets
