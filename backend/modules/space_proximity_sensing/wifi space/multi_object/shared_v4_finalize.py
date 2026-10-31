
import time,json,copy,traceback
import joblib,numpy as np
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from shared_v4_common import O4,read,atomic,lock,now,decode,STATE_COUNTS
from shared_v4_cache import load
from shared_v4_train import group_score,dump
from night_infer import component_prob

def select():
    if (O4/'frozen.json').exists():raise RuntimeError('Already frozen; no further selection allowed')
    old_selection=read(O4/'selection.json');atomic('initial_selection.json',old_selection)
    b=joblib.load(O4/'selected.joblib');old=joblib.load(O4.parent/'overnight'/'model.joblib');va=load('validation');y=va['labels'];counts=va['object_count'];reports={}
    with threadpool_limits(limits=4):
        for view,e in b['experts'].items():
            cc=e['components'];X=va[view];p=component_prob(cc['tree'],X);op=component_prob(old['experts'][view]['components']['tree'],X);rc=component_prob(cc['regions'],X);oc=component_prob(cc['objects'],X);lp=component_prob(cc['layout'],X);best=None;comparison=[]
            for w in [.75,1.]:
                mix=w*p+(1-w)*op
                for bias in [-.25,0,.25]:
                    for rw in [0,.5,1]:
                        for lw in [.25,.5,1]:
                            for ow in [.5,1,2]:
                                cfg=dict(tree_weight=w,odds_bias=bias,region_weight=rw,layout_weight=lw,object_weight=ow);pred,nobj,post=decode(mix,rc,lp,oc,cfg);gs=group_score(y,counts,pred,nobj,va['zone_count'],STATE_COUNTS[post.argmax(1)]);f1=float(f1_score(y,pred,average='macro',zero_division=0));score=(gs,f1,float(np.mean(nobj==counts)),w);row=dict(config=cfg,balanced_joint_exact=gs,zone_macro_f1=f1,object_count_accuracy=score[2]);comparison.append(row)
                                if best is None or score>best[0]:best=(score,row)
            e['decoder']=best[1]['config'];e.pop('old_tree',None)
            if e['decoder']['tree_weight']<1:e['old_tree']=old['experts'][view]['components']['tree']
            reports[view]=dict(best=best[1],comparison=sorted(comparison,key=lambda a:(a['balanced_joint_exact'],a['zone_macro_f1'],a['object_count_accuracy']),reverse=True)[:8])
    b['selection_reports']=reports;b['primary_view']=max(reports,key=lambda v:reports[v]['best']['balanced_joint_exact']);dump(b,O4/'selected.joblib');atomic('selection.json',dict(selected_at=now(),primary=b['primary_view'],reports=reports,joint_states=219,expanded_grid='Added region_weight=0 before any final scoring; model components unchanged.',test_used=False));print('Final validation-only decoder selected.',flush=True)

def main():
    with lock('finalizer'):
        if (O4/'frozen.json').exists():
            from shared_v4_evaluate import main as evaluate
            evaluate();return
        atomic('finalizer_state.json',dict(state='waiting_for_training',updated=now()))
        while not (O4/'training_state.json').exists() or read(O4/'training_state.json')['state']!='selected':
            if (O4/'training_state.json').exists() and read(O4/'training_state.json')['state']=='failed':raise RuntimeError('Training failed')
            time.sleep(5)
        select();atomic('finalizer_state.json',dict(state='evaluating_frozen_candidate',updated=now()))
        from shared_v4_evaluate import main as evaluate
        evaluate();atomic('finalizer_state.json',dict(state='completed',updated=now()))
if __name__=='__main__':
    try:main()
    except Exception as e:atomic('finalizer_state.json',dict(state='failed',updated=now(),error=str(e),traceback=traceback.format_exc()));raise
