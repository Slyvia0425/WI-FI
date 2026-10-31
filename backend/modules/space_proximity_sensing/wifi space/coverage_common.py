
import numpy as np
from learning import ROOT,CFG,features
DATA=ROOT/'data/coverage_v3'
OUT=ROOT/'outputs/coverage_v3'
BOUNDS=[.61,5.39,.41,5.59]
CLEARANCE=.08

def devices(offsets=None):
    p=np.array(CFG['tx_m']+CFG['rx_m'],dtype=float)
    if offsets is not None:p[:,:2]+=np.asarray(offsets)
    return p

def valid_placement(xy,offsets=None):
    x,y=xy
    if not np.isfinite([x,y]).all() or not(BOUNDS[0]<=x<=BOUNDS[1] and BOUNDS[2]<=y<=BOUNDS[3]):return False
    d=np.abs(devices(offsets)[:,:2]-[x,y])
    return not np.any(np.all(d<=np.array([.6,.4])+CLEARANCE,axis=1))

def near_boundary(xy,width=.15):
    return min(abs(xy[0]-2),abs(xy[0]-4),abs(xy[1]-2),abs(xy[1]-4))<width

def signal_features(h,reference):
    base=features(h,reference)
    extra=[]


    for lag in [1,8,24]:
        a=h[:,lag:];b=h[:,:-lag]
        den=np.sum(np.abs(a)*np.abs(b),axis=1)+1e-18
        c=np.sum(a*np.conj(b),axis=1)/den
        extra.extend([c.real,c.imag])
    return np.concatenate([base,*extra]).astype(np.float32)

def probabilities(bundle,X):
    p=np.clip(bundle['model'].predict_proba(X),1e-12,1)
    z=np.log(p)/bundle.get('temperature',1);z-=z.max(1,keepdims=True)
    p=np.exp(z);return p/p.sum(1,keepdims=True)

def decisions(bundle,X,p=None):
    if p is None:p=probabilities(bundle,X)
    dist=bundle['novelty_neighbors'].kneighbors(bundle['novelty_scaler'].transform(X))[0][:,0]
    masks=p>=bundle['set_cutoff']


    weak=np.flatnonzero(masks.sum(1)==0)
    if len(weak):
        top2=np.argsort(p[weak],axis=1)[:,-2:]
        masks[weak[:,None],top2]=True
    keep=(p.max(1)>=bundle['threshold'])&(dist<=bundle['novelty_limit'])&(masks.sum(1)==1)
    return p,dist,masks,keep
