
import time
import numpy as np
from night_features import views
from night_infer import component_prob,posterior,route
from layout_t1_v1_common import STATE_LAYOUT,STATE_COUNT,STATE_MASK,STATE_COUNTS,decode
from opt_v2_common import MASKS

def expert(b,view,X):
    e=b['experts'][view];cc=e['components'];p=component_prob(cc['tree'],X);w=e['decoder']['tree_weight']
    if w<1:p=w*p+(1-w)*component_prob(e['old_tree'],X)
    return decode(p,component_prob(cc['regions'],X),component_prob(cc['layout'],X),component_prob(cc['objects'],X),e['decoder'])

def refined_posterior(b,d,r=None):
    from layout_t1_v1_infer import batch as original
    if r is None:r=original(b,d)
    post=r['posterior'].copy();valid=(r['routing']=='calibrated')&r['quality']
    if valid.any():
        cfg=b['count_refiner'];parts=[]
        for item in cfg['models']:
            model=item['model'];parts.append(model.predict(d['calibrated'][valid],thread_count=2) if item['name']=='boost' else model.predict(d['calibrated'][valid]))
        predicted=np.mean(parts,axis=0)*cfg['scale'];error=((predicted[:,None,:]-STATE_COUNTS[None,:,:])**2).sum(2)
        score=np.log(np.maximum(post[valid],1e-30))-cfg['weight']*error;score-=score.max(1,keepdims=True);p=np.exp(score);post[valid]=p/p.sum(1,keepdims=True)
    return post

def batch(b,d):
    from layout_t1_v1_infer import batch as original
    r=original(b,d);r['posterior']=posterior(refined_posterior(b,d,r),b.get('refiner_temperature',1.))
    r['routing'][(r['routing']=='calibrated')&r['quality']]='count_refiner'
    r['zone_counts']=STATE_COUNTS[r['posterior'].argmax(1)];r['pred']=r['zone_counts']>0;r['counts']=r['zone_counts'].sum(1)
    r['accepted']=r['quality']&(r['posterior'].max(1)>=b['joint_gate']);return r

def infer_v5(b,h):
    start=time.perf_counter();base=dict(status='Unknown',regions=[],probabilities=[0.]*9,estimated_region_count=None,estimated_object_count=None,confirmed_regions=0,confirmed_objects=0,model_version=5)
    if np.asarray(h).ndim in [2,3] and np.asarray(h).shape[-2:]==(32,64) and np.any(np.all(np.abs(h)==0,axis=tuple(i for i in range(np.asarray(h).ndim) if i!=np.asarray(h).ndim-2))):return dict(base,reason='missing_required_links',elapsed_ms=(time.perf_counter()-start)*1000)
    try:v=views(h,b['reference'])
    except (ValueError,TypeError) as e:return dict(base,reason=str(e),elapsed_ms=(time.perf_counter()-start)*1000)
    q=np.array([[v[k] for k in ['phase_health','temporal_coherence','roughness_cal','roughness_free','calibrated_usable']]]);d={k:v[k][None] for k in ['calibrated','phase_free']};d['quality']=q;r=batch(b,d);view=str(r['routing'][0]);info=dict(frames=v['frames'],feature_view=view,signal_roughness=float(q[0,2 if view in ['calibrated','count_refiner'] else 3]),phase_health=float(q[0,0]),temporal_coherence=float(q[0,1]))
    if not r['quality'][0]:return dict(base,reason='signal_roughness_outside_calibration',elapsed_ms=(time.perf_counter()-start)*1000,**info)
    pred=r['pred'][0];nobj=int(r['counts'][0]);pp=r['posterior'][0];marg=pp@STATE_MASK;joint=float(pp.max());accepted=bool(r['accepted'][0] and v['frames']==b['frames'])
    layout_post=np.bincount(STATE_LAYOUT,weights=pp,minlength=len(MASKS));count_post=np.bincount(STATE_COUNT-1,weights=pp,minlength=3);selected_layout=int(STATE_LAYOUT[pp.argmax()])
    zone_count=r['zone_counts'][0]
    regions=[dict(zone=int(z),object_count=int(zone_count[z]),probability=float(marg[z]),threshold=b['joint_gate'],accepted=accepted) for z in np.flatnonzero(pred)]
    alternatives=[dict(zones=np.flatnonzero(STATE_MASK[i]).tolist(),per_zone_object_count=STATE_COUNTS[i].tolist(),object_count=int(STATE_COUNT[i]),probability=float(pp[i])) for i in np.argsort(pp)[-3:][::-1]]
    return dict(base,status='Accepted regions' if accepted else 'Uncertain',reason='joint_layout_and_object_count' if v['frames']==b['frames'] else 'nonstandard_frame_count_not_16_frame_validated',regions=regions,probabilities=marg.tolist(),per_zone_object_count=zone_count.tolist(),estimated_region_count=int(pred.sum()),estimated_object_count=nobj,confirmed_regions=len(regions) if accepted else 0,confirmed_objects=nobj if accepted else 0,joint_confidence=joint,layout_confidence=float(layout_post[selected_layout]),object_count_confidence=float(count_post[nobj-1]),count_probabilities=count_post.tolist(),alternative_layouts=alternatives,elapsed_ms=(time.perf_counter()-start)*1000,note='Learned object counts per occupied zone; exact within-zone individual coordinates and sizes are not recovered.',**info)
