
import hashlib,json
from pathlib import Path
import numpy as np
from opt_v2_common import extract,collapse,roughness
from night_phase import phase_health

SPEC=dict(version='night-features-1',frames=16,calibrated_features=1600,phase_free_features=1088,phase_free_preprocessing='Align each frame/link to reference correlation phase before coherent averaging; remove direct absolute-phase features.',quality='Separate phase health, frame coherence and calibrated/free roughness. These are routing/quality signals, not geometry inputs.')

def aligned_average(h,reference):
    h=np.asarray(h,np.complex64)
    if h.ndim==2:h=h[None]
    if h.ndim!=3 or h.shape[1:]!=(32,64) or not 1<=len(h)<=64 or not np.isfinite(h).all():raise ValueError('Expected finite [frames,32,64]')
    ref=np.asarray(reference,np.complex64);cross=np.sum(h*np.conj(ref[None]),axis=2,keepdims=True)
    phase=np.conj(cross)/np.maximum(np.abs(cross),1e-30);aligned=h*phase
    return aligned.mean(0).astype(np.complex64)

def temporal_coherence(h):
    h=np.asarray(h,np.complex64)
    if h.ndim==2:return 1.
    rms_frames=np.sqrt(np.mean(np.abs(h)**2,axis=2)).mean(0);rms_mean=np.sqrt(np.mean(np.abs(h.mean(0))**2,axis=1))
    good=rms_frames>max(float(rms_frames.max())*1e-3,1e-12)
    return float(np.median(rms_mean[good]/np.maximum(rms_frames[good],1e-20))) if good.any() else 0.

def views(h,reference):
    avg,frames=collapse(h)
    free=aligned_average(h,reference)
    usable=bool(np.any(np.abs(avg)>1e-12))


    phase_free=extract(free,reference,calibrated=False)
    return dict(calibrated=extract(avg,reference,calibrated=True) if usable else np.zeros(1600,np.float32),calibrated_usable=usable,phase_free=phase_free,frames=frames,phase_health=phase_health(h,reference)['score'] if usable else float(np.pi),temporal_coherence=temporal_coherence(h),roughness_cal=roughness(avg,reference) if usable else 1e6,roughness_free=roughness(free,reference))

def fingerprint(reference):
    files=[Path(__file__),Path(__file__).with_name('opt_v2_common.py'),Path(__file__).with_name('night_phase.py')]
    return hashlib.sha256(json.dumps(SPEC,sort_keys=True).encode()+np.asarray(reference,np.complex64).tobytes()+b''.join(p.read_bytes() for p in files)).hexdigest()
