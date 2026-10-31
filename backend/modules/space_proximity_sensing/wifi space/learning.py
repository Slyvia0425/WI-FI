
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))

def observe(h,rng,snr_db=None,gain_sigma=None):

    h=np.asarray(h,dtype=np.complex64)
    if snr_db is None:snr_db=rng.uniform(*CFG['noise_snr_db'])
    if gain_sigma is None:gain_sigma=CFG['gain_sigma_db']
    rms=np.sqrt(np.mean(np.abs(h)**2,axis=1,keepdims=True))
    noise=(rng.normal(size=h.shape)+1j*rng.normal(size=h.shape))*rms*10**(-snr_db/20)/np.sqrt(2)
    gain=10**(rng.normal(0,gain_sigma,(h.shape[0],1))/20)
    return ((h+noise)*gain).astype(np.complex64)

def features(h,reference):
    h=np.asarray(h)
    if h.shape!=(12,64):raise ValueError('Expected complex CSI array [12 links, 64 subcarriers].')
    if not np.isfinite(h).all():raise ValueError('Non-finite signal: Unknown.')
    amp=np.abs(h)
    if not np.any(amp>1e-12):raise ValueError('No usable signal: Unknown.')
    db=20*np.log10(np.maximum(amp,1e-10))
    refdb=20*np.log10(np.maximum(np.abs(reference),1e-10))
    delta=np.clip(db-refdb,-60,60)

    band=delta.reshape(12,8,8).mean(2)
    x=np.concatenate([band.flatten(),db.mean(1),db.std(1),np.mean(np.abs(np.diff(db,axis=1)),axis=1)])
    return x.astype(np.float32)

def zone_center(zone):return [float(1+2*(zone%3)),float(1+2*(zone//3))]

def classify(bundle,h):
    if bundle.get('protocol')==3:
        from coverage_infer import classify_coverage
        return classify_coverage(bundle,h)
    import time
    t=time.perf_counter()
    try:
        x=features(h,bundle['reference'])
    except ValueError as e:
        return dict(prediction=None,confidence=0,probabilities=[0.0]*9,reason=str(e),center_m=None,elapsed_ms=(time.perf_counter()-t)*1000)
    p=bundle['model'].predict_proba(x[None])[0]
    index=int(np.argmax(p));confidence=float(p[index]);accepted=confidence>=bundle['threshold'];reason='accepted' if accepted else 'low_confidence';distance=None
    if 'novelty_neighbors' in bundle:
        distance=float(bundle['novelty_neighbors'].kneighbors(bundle['novelty_scaler'].transform(x[None]))[0][0,0])
        if distance>bundle['novelty_limit']:accepted=False;reason='outside_signal_reference_range'
    return dict(prediction=index if accepted else None,raw_prediction=index,confidence=confidence,probabilities=p.tolist(),reason=reason,novelty_distance=distance,center_m=zone_center(index) if accepted else None,elapsed_ms=(time.perf_counter()-t)*1000)
