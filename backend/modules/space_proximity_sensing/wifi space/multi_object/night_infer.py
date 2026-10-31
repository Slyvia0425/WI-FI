
import time
import numpy as np
from opt_v2_common import MASKS,decode_scores
from night_features import views

def component_prob(c,X):
    if 'scaler' in c:X=c['scaler'].transform(X)
    if c['kind']=='tree':return np.asarray(c['model'].predict_proba(X,thread_count=2))
    return np.asarray(c['model'].predict_proba(X))

def expert_prob(b,view,X):
    e=b['experts'][view];cc=e['components'];cfg=e['decoder']
    w=cfg['tree_weight']
    if w==1:p=component_prob(cc['tree'],X)
    else:
        p=component_prob(cc['occ'],X) if cfg['net_source']!='v2' else None
        if cfg['net_source']!='new':
            old=b['old_occupancy'];xx=old['scaler'].transform(X)
            op=np.mean([m.predict_proba(xx) for m in old['models']],axis=0)
            p=op if cfg['net_source']=='v2' else (op+p)/2
        if w:p=(1-w)*p+w*component_prob(cc['tree'],X)
    p=np.clip(p,1e-7,1-1e-7);p=1/(1+np.exp(-(np.log(p/(1-p))+cfg['odds_bias'])))
    c=component_prob(cc['count'],X);l=component_prob(cc['layout'],X)
    pred,post,_=decode_scores(p,c,l,cfg['count_weight'],cfg['layout_weight'])
    return pred,post,c

def posterior(post,temperature):
    z=np.log(np.maximum(post,1e-20))/temperature;z-=z.max(1,keepdims=True)
    pp=np.exp(z);return pp/pp.sum(1,keepdims=True)

def route(b,quality):

    q=np.asarray(quality);out=np.full(len(q),b['primary_view'],dtype=object)
    if 'phase_free' in b['experts']:
        bad=(q[:,0]>b['phase_limit'])|(q[:,1]<b['coherence_limit'])|(q[:,4]<.5)
        out[bad]='phase_free'
    return out

def batch(b,data):
    n=len(data['quality']);routing=route(b,data['quality']);pred=np.zeros((n,9),bool);pp=np.zeros((n,len(MASKS)));cp=np.zeros((n,3));quality=np.zeros(n,bool)
    for view in b['experts']:
        ix=np.flatnonzero(routing==view)
        if not len(ix):continue
        p,post,c=expert_prob(b,view,data[view][ix]);pred[ix]=p;cp[ix]=c;pp[ix]=posterior(post,b['experts'][view]['temperature'])
        rough=data['quality'][ix,2 if view=='calibrated' else 3]
        quality[ix]=rough<=b['experts'][view]['roughness_limit']
    accepted=quality&(pp.max(1)>=b['layout_gate'])
    return dict(pred=pred,posterior=pp,count_probabilities=cp,quality=quality,accepted=accepted,routing=routing)

def infer_v3(b,h):
    start=time.perf_counter();base=dict(status='Unknown',regions=[],probabilities=[0.]*9,estimated_region_count=None,confirmed_regions=0,model_version=3)
    try:v=views(h,b['reference'])
    except (ValueError,TypeError) as e:return dict(base,reason=str(e),elapsed_ms=(time.perf_counter()-start)*1000)
    q=np.array([[v[k] for k in ['phase_health','temporal_coherence','roughness_cal','roughness_free','calibrated_usable']]])
    d={k:v[k][None] for k in ['calibrated','phase_free']};d['quality']=q;r=batch(b,d);view=str(r['routing'][0]);rough=float(q[0,2 if view=='calibrated' else 3])
    info=dict(frames=v['frames'],signal_roughness=rough,feature_view=view,phase_health=float(q[0,0]),temporal_coherence=float(q[0,1]),elapsed_ms=(time.perf_counter()-start)*1000)
    if not r['quality'][0]:return dict(base,reason='signal_roughness_outside_calibration',**info)
    pred=r['pred'][0];pp=r['posterior'][0];marg=pp@MASKS;confidence=float(pp.max());accepted=bool(r['accepted'][0] and v['frames']==b['frames'])
    regions=[dict(zone=int(z),probability=float(marg[z]),threshold=b['layout_gate'],accepted=accepted) for z in np.flatnonzero(pred)]
    alternatives=[dict(zones=np.flatnonzero(MASKS[i]).tolist(),probability=float(pp[i])) for i in np.argsort(pp)[-3:][::-1]]
    return dict(base,status='Accepted regions' if accepted else 'Uncertain',reason='structured_signal_layout' if v['frames']==b['frames'] else 'nonstandard_frame_count_not_16_frame_validated',regions=regions,probabilities=marg.tolist(),estimated_region_count=int(pred.sum()),confirmed_regions=len(regions) if accepted else 0,layout_confidence=confidence,count_probabilities=r['count_probabilities'][0].tolist(),alternative_layouts=alternatives,note='Fixed-room simulated center-zone classification. Phase-free routing is not proof of hardware or room generalization.',**info)
