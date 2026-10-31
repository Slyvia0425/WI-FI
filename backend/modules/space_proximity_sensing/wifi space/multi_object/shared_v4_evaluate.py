
import os,time,json,hashlib,traceback
from pathlib import Path
import joblib,numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score,recall_score,confusion_matrix,multilabel_confusion_matrix
from threadpoolctl import threadpool_limits
from shared_v4_common import O4,D4,atomic,read,lock,now,sha,STATE_LAYOUT,STATE_COUNT,STATE_MASK,state_ids
from shared_v4_cache import load,build,jobs_for
from shared_v4_infer import expert,batch,infer_v4
from shared_v4_train import dump
from night_infer import batch as old_batch,posterior

def simple_metrics(y,c,p,pc):
    layout=np.all(y==p,axis=1);count=c==pc
    return dict(n=len(y),occupied_zone_macro_f1=float(f1_score(y,p,average='macro',zero_division=0)),count_accuracy=float(count.mean()),region_count_accuracy=float(np.mean(y.sum(1)==p.sum(1))),exact_occupied_layout_accuracy=float(layout.mean()),exact_joint_accuracy=float((layout&count).mean()))

def metrics(d,r):
    y=d['labels'];c=d['object_count'];p=r['pred'];pc=r['counts'];m=simple_metrics(y,c,p,pc);joint=np.all(d['zone_count']==r['zone_counts'],axis=1);m['exact_joint_accuracy']=float(joint.mean());keep=r['accepted'];quality=r['quality'];shared=c>y.sum(1)
    m.update(quality_coverage=float(quality.mean()),accepted_layout_coverage=float(keep.mean()),accepted_layout_accuracy=float(joint[keep].mean()) if keep.any() else None,accepted_joint_accuracy=float(joint[keep].mean()) if keep.any() else None,accepted_n=int(keep.sum()),all_inputs_operational_exact_accuracy=float(np.mean(joint&quality)),per_zone_recall=recall_score(y,p,average=None,zero_division=0).tolist(),per_zone_confusion=multilabel_confusion_matrix(y,p).tolist(),object_count_confusion=confusion_matrix(c,pc,labels=[1,2,3]).tolist())
    m['by_true_count']={str(n):simple_metrics(y[c==n],c[c==n],p[c==n],pc[c==n]) for n in [1,2,3]}
    for n in [1,2,3]:m['by_true_count'][str(n)]['exact_joint_accuracy']=float(joint[c==n].mean())
    m['by_sharing']={}
    for key,ix in [('distinct',~shared),('shared',shared),('all_in_one_zone',(c>1)&(y.sum(1)==1))]:
        if not ix.any():continue
        mm=simple_metrics(y[ix],c[ix],p[ix],pc[ix]);kk=ix&keep;mm.update(exact_joint_accuracy=float(joint[ix].mean()),accepted_coverage=float(kk.sum()/ix.sum()),accepted_joint_accuracy=float(joint[kk].mean()) if kk.any() else None);m['by_sharing'][key]=mm
    return m

def calibration():
    while not (O4/'selection.json').exists():
        p=O4/'training_state.json'
        if p.exists() and read(p)['state']=='failed':raise RuntimeError('Training failed')
        time.sleep(5)
    b=joblib.load(O4/'selected.joblib');d=load('calibration',True);ids=state_ids(d['zone_count']);b.update(phase_limit=float(max(np.quantile(d['quality'][:,0],.995)+.05,.1)),coherence_limit=float(max(0,np.quantile(d['quality'][:,1],.005)-.05)))
    for view,e in b['experts'].items():
        p,c,post=expert(b,view,d[view])
        def loss(t):return -np.log(np.maximum(posterior(post,t)[np.arange(len(ids)),ids],1e-20)).mean()
        e['temperature']=float(minimize_scalar(loss,bounds=(.5,4),method='bounded').x);e['roughness_limit']=float(max(np.quantile(d['quality'][:,2 if view=='calibrated' else 3],.995)*1.5,.08))
    b['joint_gate']=.95;r=batch(b,d);correct=np.all(d['zone_count']==r['zone_counts'],axis=1);choices=[]
    for gate in np.linspace(.35,.99,65):
        k=r['quality']&(r['posterior'].max(1)>=gate)
        if k.sum()>=40 and correct[k].mean()>=.9:choices.append(float(gate))
    b['joint_gate']=min(choices) if choices else .99;b['calibration_report']=dict(joint_gate=b['joint_gate'],target_met=bool(choices),metrics=metrics(d,batch(b,d)))
    dump(b,O4/'model.joblib');atomic('frozen.json',dict(frozen_at=now(),model_sha256=sha(O4/'model.joblib'),source_sha256={n:sha(Path(__file__).with_name(n)) for n in ['shared_v4_infer.py','shared_v4_common.py','night_features.py','night_infer.py']},selection_sha256=sha(O4/'selection.json'),release_rule='Shared physical-count accuracy>=.65, shared zone Macro-F1>=.75, shared joint accuracy>=.45, distinct layout accuracy at most5 percentage points below v3 on SAME new scenes, plus data/API checks. Test is release check, never tuning.',test_opened=False));return b

def main():
    with lock('evaluation'),threadpool_limits(limits=4):
        atomic('evaluation_state.json',dict(state='waiting_for_selection',pid=os.getpid(),updated=now()))
        if (O4/'frozen.json').exists():assert sha(O4/'model.joblib')==read(O4/'frozen.json')['model_sha256'];b=joblib.load(O4/'model.joblib')
        else:b=calibration()
        old=joblib.load(O4.parent/'overnight'/'model.joblib');report=dict(version=4,selected_model=b['name'],model_bytes=(O4/'model.joblib').stat().st_size,training=dict(physical_scenes=48000,new_shared_scenes=12000,old_training_only=36000,observations=96000),split_counts=dict(train=48000,validation=1200,calibration=800,test=1800,boundary_test=450),selection='New validation for checkpoints and219-state decoder; separate800 calibration for confidence; new final tests sealed until frozen.json. No physical count/positions supplied to inference.',splits={},baseline_comparison={},comparison=[],calibration=b['calibration_report'],target='Occupied center zones and physical object count are separate; several objects may share a center zone.')
        for split in ['validation','calibration','test','boundary_test']:
            if split in ['test','boundary_test']:
                if not (O4/'test_opened.json').exists():atomic('test_opened.json',dict(opened_at=now(),frozen_sha256=read(O4/'frozen.json')['model_sha256']))
                build(split)
            d=load(split,True);r=batch(b,d);br=old_batch(old,d);br['counts']=br['pred'].sum(1).astype(int);br['zone_counts']=br['pred'].astype(int);report['splits'][split]=metrics(d,r);report['baseline_comparison'][split]={'v3_region_count_as_object_count':metrics(d,br)}
            np.savez_compressed(O4/('predictions_'+split+'.npz'),ids=d['ids'],truth=d['labels'],true_object_count=d['object_count'],true_zone_count=d['zone_count'],pred=r['pred'],pred_zone_count=r['zone_counts'],pred_object_count=r['counts'],posterior=r['posterior'],quality=r['quality'],accepted=r['accepted']);atomic('evaluation_state.json',dict(state='evaluating',pid=os.getpid(),updated=now(),split=split,metrics=report['splits'][split]));print(split,report['splits'][split],flush=True)
        s=report['splits']['test']['by_sharing'];o=report['baseline_comparison']['test']['v3_region_count_as_object_count']['by_sharing'];checks=dict(shared_object_count=s['shared']['count_accuracy']>=.65,shared_occupancy=s['shared']['occupied_zone_macro_f1']>=.75,shared_joint=s['shared']['exact_joint_accuracy']>=.45,distinct_layout_noninferior=s['distinct']['exact_occupied_layout_accuracy']>=o['distinct']['exact_occupied_layout_accuracy']-.05)
        report['release_check']=dict(eligible=all(checks.values()),checks=checks)
        from opt_v2_common import window
        folder,row=jobs_for('test')[0]
        with np.load(folder/'records'/(row['id']+'.npz')) as a:h=window(a['h'],np.random.default_rng(2026103016),30)
        times=[]
        for i in range(35):
            st=time.perf_counter();infer_v4(b,h)
            if i>=5:times.append((time.perf_counter()-st)*1000)
        report['inference_ms']=dict(median=float(np.median(times)),p95=float(np.percentile(times,95)),scope='Signal-only features/classification/confidence; excludes RT and acquisition')
        atomic('results.json',report);atomic('evaluation_state.json',dict(state='completed',pid=os.getpid(),updated=now(),release=report['release_check']));print('Release',report['release_check'],flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:atomic('evaluation_state.json',dict(state='failed',pid=os.getpid(),updated=now(),error=str(e),traceback=traceback.format_exc()));raise
