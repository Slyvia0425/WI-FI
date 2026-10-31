
import numpy as np
from pathlib import Path
from itertools import combinations
from multi_common import ROOT,DATA,OUT,noisy
DATA2=DATA/'optimized_v2';OUT2=OUT/'optimized_v2'
MASKS=np.array([np.isin(np.arange(9),c) for n in [1,2,3] for c in combinations(range(9),n)],dtype=np.float32)
BITS=(MASKS.astype(np.int64)*(1<<np.arange(9))).sum(1)
MASK_INDEX={int(v):i for i,v in enumerate(BITS)}

def targets(rows):
    y=np.zeros((len(rows),9),np.float32)
    for i,row in enumerate(rows):
        for obj in row['objects']:y[i,obj['zone']]=1
    return y

def window(h,rng,snr_db=None,frames=16):

    if snr_db is None:snr_db=float(rng.uniform(27,40))
    return np.stack([noisy(h,rng,snr_db) for _ in range(frames)])

def collapse(h):
    h=np.asarray(h,dtype=np.complex64)
    if h.ndim==3:
        if h.shape[1:]!=(32,64) or not 1<=h.shape[0]<=64:raise ValueError('Expected [frames,32,64], with 1–64 frames.')
        if not np.isfinite(h).all():raise ValueError('Nonfinite CSI frames')
        return h.mean(0).astype(np.complex64),int(h.shape[0])
    if h.shape!=(32,64) or not np.isfinite(h).all():raise ValueError('Expected [32,64] or [frames,32,64].')
    return h,1

def extract(h,reference,calibrated=True):
    h,_=collapse(h)
    if not np.any(np.abs(h)>1e-12):raise ValueError('No usable signal')
    ref=np.asarray(reference,np.complex64);ref_rms=np.sqrt(np.mean(np.abs(ref)**2,axis=1,keepdims=True))
    floor=np.maximum(ref_rms*10**(-45/20),1e-10)
    amp=np.abs(h);refamp=np.abs(ref)
    delta=np.clip(20*np.log10(np.maximum(amp,floor)/np.maximum(refamp,floor)),-45,25)
    blocked=(amp<floor).mean(1)
    parts=[delta.reshape(32,8,8).mean(2).ravel(),delta.mean(1),delta.std(1),np.abs(np.diff(delta,axis=1)).mean(1),blocked]
    cross=np.sum(h*np.conj(ref),axis=1,keepdims=True)
    align=np.conj(cross)/np.maximum(np.abs(cross),1e-20)
    residual=(h*align-ref)/np.maximum(ref_rms,1e-10)
    r=np.clip(residual.real,-6,6).reshape(32,8,8).mean(2)
    im=np.clip(residual.imag,-6,6).reshape(32,8,8).mean(2)
    parts.extend([r.ravel(),im.ravel()])
    if calibrated:


        direct=(h-ref)/np.maximum(ref_rms,1e-10)
        parts.extend([np.clip(direct.real,-6,6).reshape(32,8,8).mean(2).ravel(),np.clip(direct.imag,-6,6).reshape(32,8,8).mean(2).ravel()])
    for lag in [1,8,24]:
        a=h[:,lag:];b=h[:,:-lag];den=np.maximum(np.sum(np.abs(a)*np.abs(b),axis=1),((64-lag)*floor[:,0]**2))
        corr=np.sum(a*np.conj(b),axis=1)/np.maximum(den,1e-20);parts.extend([corr.real,corr.imag])
    return np.concatenate(parts).astype(np.float32)

def roughness(h,reference):
    h,_=collapse(h);power=np.mean(np.abs(h)**2,axis=1);refpower=np.mean(np.abs(reference)**2,axis=1)
    valid=power>refpower*1e-4
    if not valid.any():return 1e6
    score=np.sqrt(np.mean(np.abs(np.diff(h,axis=1))**2,axis=1)/np.maximum(power,1e-20))
    return float(np.median(score[valid]))

def layout_ids(y):return np.array([MASK_INDEX[int(np.sum(row.astype(int)*(1<<np.arange(9))))] for row in y])

def decode_scores(p,count_p,layout_p=None,count_weight=1.,layout_weight=0.):
    p=np.clip(p,1e-7,1-1e-7)
    score=np.log(p)@MASKS.T+np.log1p(-p)@(1-MASKS).T
    if count_weight:score+=count_weight*np.log(np.maximum(count_p[:,MASKS.sum(1).astype(int)-1],1e-12))
    if layout_p is not None and layout_weight:score+=layout_weight*np.log(np.maximum(layout_p,1e-12))
    score-=score.max(1,keepdims=True);posterior=np.exp(score);posterior/=posterior.sum(1,keepdims=True)
    labels=posterior.argmax(1);return MASKS[labels],posterior,posterior@MASKS
