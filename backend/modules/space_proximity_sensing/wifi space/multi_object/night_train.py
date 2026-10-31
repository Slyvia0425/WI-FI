
import os,time,json,copy,hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from night_common import NIGHT,atomic,stage_lock,log,now
from night_cache import load
from night_learn import task,predict,safe_dump
from opt_v2_common import OUT2,decode_scores
from opt_v2_train import raw_prob,metrics,balanced_exact

def adjust(p,bias):
    p=np.clip(p,1e-7,1-1e-7);return 1/(1+np.exp(-(np.log(p/(1-p))+bias)))

def select(components):
    va=load('validation');y=va['labels'];old=joblib.load(OUT2/'model.joblib');experts={};reports={};baselines={}
    with threadpool_limits(limits=4):
        oldp,oldc,oldl=raw_prob(old,va['calibrated']);bp,_,_=decode_scores(adjust(oldp,old['odds_bias']),oldc,oldl,old['count_weight'],old['layout_weight']);baselines['v2']=metrics(y,bp)
        for view in ['calibrated','phase_free']:
            needed=[view+'_'+k for k in ['occ','count','layout']]
            if any(k not in components for k in needed):reports[view]=dict(state='incomplete',missing=[k for k in needed if k not in components]);continue
            cc={k:joblib.load(components[view+'_'+k]['path']) for k in ['occ','count','layout']};X=va[view]
            newp=predict(cc['occ'],X);cp=predict(cc['count'],X);lp=predict(cc['layout'],X)
            nets=[('new',newp)]
            if view=='calibrated':nets.extend([('v2',oldp),('old_new_mean',(oldp+newp)/2)])
            tp=None
            if view+'_tree' in components:cc['tree']=joblib.load(components[view+'_tree']['path']);tp=predict(cc['tree'],X)
            possibilities=[];best=None
            for source,np_ in nets:
                for tw in ([0,.25,.5,.75,1] if tp is not None else [0]):
                    p=(1-tw)*np_+tw*tp if tp is not None else np_
                    for bias in [-.25,0,.25]:
                        pp=adjust(p,bias)
                        for cw in [.5,1,2]:
                            for lw in [.25,.5,1]:
                                pred,_,_=decode_scores(pp,cp,lp,cw,lw);m=metrics(y,pred);score=(balanced_exact(y,pred),m['occupied_zone_macro_f1'],-tw)
                                cfg=dict(net_source=source,tree_weight=tw,odds_bias=bias,count_weight=cw,layout_weight=lw);r=dict(config=cfg,balanced_two_three_exact=score[0],validation=m);possibilities.append(r)
                                if best is None or score>best[0]:best=(score,r)
            reports[view]=dict(state='selected',best=best[1],comparison=sorted(possibilities,key=lambda r:(r['balanced_two_three_exact'],r['validation']['occupied_zone_macro_f1']),reverse=True)[:8]);experts[view]=dict(components=cc,decoder=best[1]['config'])


    old_occ=dict(scaler=old['scaler'],models=old['models'])
    for m in old_occ['models']:
        if hasattr(m,'_optimizer'):del m._optimizer
    if not experts:raise RuntimeError('No complete expert')
    candidate=dict(version=3,name='Expanded-data CSI ensemble with phase-free expert',experts=experts,old_occupancy=old_occ,reference=old['reference'],frames=16,selection_reports=reports,baseline_validation=baselines['v2'],train_cache=load('train','phase_free')['meta']['fingerprint'],validation_cache=va['meta']['fingerprint'])
    path=NIGHT/'candidate_uncalibrated.joblib';safe_dump(candidate,path)
    atomic('selection.json',dict(state='selected_uncalibrated',selected_at=now(),experts=reports,baseline=baselines['v2'],candidate_path=str(path),model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),test_used=False))
    log('Fresh-validation model selection complete; calibration and final test still pending.')

def main():
    with stage_lock('training_coordinator'):
        if (NIGHT/'selection.json').exists():log('Model selection already complete; use existing versioned candidate.');return
        atomic('training_state.json',dict(state='waiting_for_cache',pid=os.getpid(),updated=now()))
        load('train','phase_free',wait=True);load('validation','phase_free',wait=True);results=[]
        names=[v+'_'+k for k in ['count','layout','occ'] for v in ['calibrated','phase_free']]
        atomic('training_state.json',dict(state='training',pid=os.getpid(),updated=now(),cpu_components=names))
        with ProcessPoolExecutor(max_workers=3) as pool:
            fs=[pool.submit(task,name) for name in names]

            for view in ['calibrated','phase_free']:results.append(task(view+'_tree'))
            results.extend(f.result() for f in fs)
        atomic('components.json',dict(results=results,finished=now()));components={r['name']:r for r in results if 'path' in r};select(components)
        atomic('training_state.json',dict(state='selected_uncalibrated',pid=os.getpid(),updated=now(),completed_components=list(components),failed_components=[r for r in results if 'error' in r]))

if __name__=='__main__':
    try:main()
    except Exception as e:
        import traceback
        atomic('training_state.json',dict(state='failed',pid=os.getpid(),updated=now(),error=str(e),traceback=traceback.format_exc()));raise
