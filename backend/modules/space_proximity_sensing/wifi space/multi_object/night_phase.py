
import os,time,json
import numpy as np
from night_common import NIGHT,atomic,stage_lock,log,now
from opt_v2_common import DATA,DATA2,collapse,window
from multi_train import seed

def phase_health(h,reference):
    h,_=collapse(h);ref=np.asarray(reference,np.complex64)
    cross=np.sum(h*np.conj(ref),axis=1);hp=np.sum(np.abs(h)**2,axis=1);rp=np.sum(np.abs(ref)**2,axis=1)
    if not np.any(hp>1e-20):raise ValueError('No usable signal')
    coherence=np.abs(cross)/np.sqrt(np.maximum(hp*rp,1e-30));ratio=np.sqrt(np.maximum(hp,1e-30)/np.maximum(rp,1e-30))
    weights=coherence**2*np.exp(-2*np.abs(np.log(np.maximum(ratio,1e-15))));angles=np.abs(np.angle(cross));order=np.argsort(angles);cumulative=np.cumsum(weights[order]);total=cumulative[-1]
    if total<1e-10:raise ValueError('No stable phase-reference links')
    values=[float(angles[order[min(len(order)-1,np.searchsorted(cumulative,q*total))]]) for q in [.25,.5,.75]]
    return dict(score=values[0],weighted_phase_quantiles=values,reference_weight=float(total),usable_links=int(np.sum(weights>.1)))

def pilot():
    with stage_lock('phase_pilot'):
        if (NIGHT/'phase_pilot.json').exists():return
        atomic('phase_state.json',dict(state='running',pid=os.getpid(),started=now()));ref=np.load(DATA/'empty_reference.npy');rows=json.loads((DATA2/'manifest.json').read_text());scores={};healthy=[]


        for split in ['calibration','validation']:
            values=[]
            for r in [r for r in rows if r['split']==split]:
                with np.load(DATA2/'records'/(r['id']+'.npz')) as a:h=a['h']
                obs=window(h,np.random.default_rng(seed('optimized-v2-'+r['id'])));values.append(phase_health(obs,ref)['score'])
                if split=='validation':healthy.append((r,h))
            scores[split]=np.array(values)
        threshold=float(max(np.quantile(scores['calibration'],.995)+.05,.10))
        conditions={};sample=healthy[:180]
        for mode in ['independent_uniform_phase','common_one_radian','small_random_phase']:
            values=[]
            for i,(r,h) in enumerate(sample):
                rng=np.random.default_rng(seed('phase-health-'+r['id']));obs=window(h,rng,30)
                phase=rng.uniform(-np.pi,np.pi,(1,32,1)) if mode=='independent_uniform_phase' else np.ones((1,32,1)) if mode=='common_one_radian' else rng.normal(0,.15,(1,32,1))
                values.append(phase_health(obs*np.exp(1j*phase),ref)['score'])
            values=np.array(values);conditions[mode]=dict(n=len(values),rejection_rate=float(np.mean(values>threshold)),score_quantiles=np.quantile(values,[0,.5,.95,1]).tolist())
        result=dict(hypothesis='Weighted lower-quarter phase residual against empty-room reference. Calibrated same-device static scene; not a correction or universal OOD detector.',threshold=threshold,threshold_source='Healthy v2 calibration only, p99.5 + .05 rad, floor .10 rad. v2 validation and synthetic drift are diagnostics, not new final test.',healthy={s:dict(n=len(v),false_rejection_rate=float(np.mean(v>threshold)),quantiles=np.quantile(v,[0,.5,.95,.995,1]).tolist()) for s,v in scores.items()},drift_diagnostics=conditions,test_used=False,release_requirement='Refit threshold on fresh calibration and measure normal false rejection plus drift rejection on fresh held-out signals before enabling.')
        atomic('phase_pilot.json',result);atomic('phase_state.json',dict(state='completed',pid=os.getpid(),finished=now(),threshold=threshold,validation_false_rejection=result['healthy']['validation']['false_rejection_rate']));log('Phase-health pilot complete; healthy validation false rejection='+str(result['healthy']['validation']['false_rejection_rate']))
if __name__=='__main__':pilot()
