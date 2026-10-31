
import time
import numpy as np
from night_features import views
from night_infer import posterior
from layout_t1_v1_common import STATE_COUNTS,STATE_COUNT,STATE_MASK,STATE_LAYOUT
from opt_v2_common import MASKS

def zone_prob(p):
 p=np.clip(np.asarray(p).reshape(-1,9,3),1e-6,1-1e-6);p=np.minimum.accumulate(p,axis=2)
 return np.clip(np.stack([1-p[:,:,0],p[:,:,0]-p[:,:,1],p[:,:,1]-p[:,:,2],p[:,:,2]],2),1e-7,1)
def ordinal_post(p,base,w):
 q=zone_prob(p);logq=np.zeros((len(p),219))
 for z in range(9):logq+=np.log(q[:,z,STATE_COUNTS[:,z]])
 z=np.log(np.maximum(base,1e-30))+w*logq;z-=z.max(1,keepdims=True);pp=np.exp(z);return pp/pp.sum(1,keepdims=True)
def base_batch(b,d):
 if b['layout_id']=='default':from home_v5_infer import batch as fn
 else:from layout_t1_v3_infer import batch as fn
 return fn(b['base_bundle'],d)
def batch(b,d,calibrate=False):
 r=base_batch(b,d);raw=r['posterior'].copy();valid=(r['routing']!='phase_free')&r['quality'];agreement=np.ones(len(raw),bool)
 if valid.any():
  X=np.column_stack([d['calibrated'][valid],np.full((valid.sum(),1),0 if b['layout_id']=='default' else 4,np.float32)])
  nn=b['ordinal_refiner']['neural'];pn=nn['model'].predict_proba(nn['scaler'].transform(X));pb=b['ordinal_refiner']['boost'].predict_proba(X,thread_count=2)
  mix=b['choice']['boost_weight'];w=b['choice']['ordinal_weight'];raw[valid]=ordinal_post(mix*pb+(1-mix)*pn,raw[valid],w)
  agreement[valid]=(ordinal_post(pn,r['posterior'][valid],w).argmax(1)==ordinal_post(pb,r['posterior'][valid],w).argmax(1))
 r['posterior']=posterior(raw,b.get('temperature',1));r['zone_counts']=STATE_COUNTS[r['posterior'].argmax(1)];r['pred']=r['zone_counts']>0;r['counts']=r['zone_counts'].sum(1)
 r['accepted']=r['quality']&(r['posterior'].max(1)>=b.get('joint_gate',1.))
 if b.get('require_agreement',True):r['accepted']&=agreement
 r['witness_agreement']=agreement;r['raw_posterior']=raw
 r['routing']=r['routing'].astype('<U24');r['routing'][valid]='ordinal_count_v6'
 return r

def infer_v5(b,h):
 start=time.perf_counter();empty=dict(status='Unknown',regions=[],probabilities=[0.]*9,estimated_region_count=None,estimated_object_count=None,confirmed_regions=0,confirmed_objects=0,model_version=5)
 h=np.asarray(h)
 if h.ndim in [2,3] and h.shape[-2:]==(32,64) and np.any(np.all(np.abs(h)==0,axis=tuple(i for i in range(h.ndim) if i!=h.ndim-2))):return dict(empty,reason='missing_required_links',elapsed_ms=(time.perf_counter()-start)*1000)
 try:v=views(h,b['reference'])
 except (ValueError,TypeError) as e:return dict(empty,reason=str(e),elapsed_ms=(time.perf_counter()-start)*1000)
 q=np.array([[v[k] for k in ['phase_health','temporal_coherence','roughness_cal','roughness_free','calibrated_usable']]]);d={k:v[k][None] for k in ['calibrated','phase_free']};d['quality']=q;r=batch(b,d);view=str(r['routing'][0]);info=dict(frames=v['frames'],feature_view=view,signal_roughness=float(q[0,3 if view=='phase_free' else 2]),phase_health=float(q[0,0]),temporal_coherence=float(q[0,1]),expert_agreement=bool(r['witness_agreement'][0]))
 if not r['quality'][0]:return dict(empty,reason='signal_roughness_outside_calibration',elapsed_ms=(time.perf_counter()-start)*1000,**info)
 zc=r['zone_counts'][0];pp=r['posterior'][0];marg=pp@STATE_MASK;joint=float(pp.max());n=int(zc.sum());accepted=bool(r['accepted'][0] and v['frames']==b['frames']);countpost=np.bincount(STATE_COUNT-1,weights=pp,minlength=3);lp=np.bincount(STATE_LAYOUT,weights=pp,minlength=len(MASKS))
 regions=[dict(zone=int(z),object_count=int(zc[z]),probability=float(marg[z]),threshold=b['joint_gate'],accepted=accepted) for z in np.flatnonzero(zc)]
 alternatives=[dict(zones=np.flatnonzero(STATE_MASK[i]).tolist(),per_zone_object_count=STATE_COUNTS[i].tolist(),object_count=int(STATE_COUNT[i]),probability=float(pp[i])) for i in np.argsort(pp)[-3:][::-1]]
 reason='ordinal_zone_counts' if accepted else 'expert_disagreement' if not info['expert_agreement'] else 'joint_confidence_below_gate'
 if v['frames']!=b['frames']:reason='nonstandard_frame_count_not_16_frame_validated'
 return dict(empty,status='Accepted regions' if accepted else 'Uncertain',reason=reason,regions=regions,probabilities=marg.tolist(),per_zone_object_count=zc.tolist(),estimated_region_count=len(regions),estimated_object_count=n,confirmed_regions=len(regions) if accepted else 0,confirmed_objects=n if accepted else 0,joint_confidence=joint,layout_confidence=float(lp[STATE_LAYOUT[pp.argmax()]]),object_count_confidence=float(countpost[n-1]),count_probabilities=countpost.tolist(),alternative_layouts=alternatives,elapsed_ms=(time.perf_counter()-start)*1000,note='Signal-derived ordinal counts in nine zones; object shapes and exact coordinates are not recovered.',**info)
