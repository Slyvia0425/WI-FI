
import os,time,json,hashlib,traceback
from pathlib import Path
import joblib,numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import recall_score,multilabel_confusion_matrix,confusion_matrix
from threadpoolctl import threadpool_limits
from night_common import NIGHT,atomic,log,now,stage_lock,alive
from night_cache import load,build,jobs_for,QUALITY
from night_features import views
from night_infer import expert_prob,posterior,route,batch,infer_v3
from night_learn import safe_dump
from opt_v2_common import OUT2,MASKS,layout_ids,window,decode_scores
from opt_v2_train import metrics,raw_prob,balanced_exact

def adjust(p,bias):
    p=np.clip(p,1e-7,1-1e-7);return 1/(1+np.exp(-(np.log(p/(1-p))+bias)))

def enrich(y,r):
    pred=r['pred'];m=metrics(y,pred);correct=np.all(y==pred,axis=1);quality=r['quality'];keep=r['accepted'];operational=pred.copy();operational[~quality]=0
    m.update(per_zone_recall=recall_score(y,pred,average=None,zero_division=0).tolist(),per_zone_confusion=multilabel_confusion_matrix(y,pred).tolist(),count_confusion=confusion_matrix(y.sum(1),pred.sum(1),labels=[1,2,3]).tolist(),quality_coverage=float(quality.mean()),accepted_layout_coverage=float(keep.mean()),accepted_layout_accuracy=float(correct[keep].mean()) if keep.any() else None,accepted_layout_n=int(keep.sum()),all_inputs_operational_exact_accuracy=float(np.mean(correct&quality)),all_inputs_operational_macro_f1=metrics(y,operational)['occupied_zone_macro_f1'])
    return m

def baseline(b,d):
    p,c,l=raw_prob(b,d['calibrated']);pred,post,_=decode_scores(adjust(p,b['odds_bias']),c,l,b['count_weight'],b['layout_weight']);pp=posterior(post,b['temperature']);q=d['quality'][:,2]<=b['roughness_limit']
    return dict(pred=pred,posterior=pp,quality=q,accepted=q&(pp.max(1)>=b['layout_gate']))

def calibrate():
    source=NIGHT/'candidate_uncalibrated.joblib'
    while not (NIGHT/'selection.json').exists():
        p=NIGHT/'training_state.json'
        if p.exists():
            state=json.loads(p.read_text(encoding='utf-8'))
            if state['state']=='failed':raise RuntimeError('Training failed: '+state.get('error',''))
        time.sleep(5)
    b=joblib.load(source);cal=load('calibration');y=cal['labels'];tid=layout_ids(y)
    primary=max(b['experts'],key=lambda v:(b['selection_reports'][v]['best']['balanced_two_three_exact'],b['selection_reports'][v]['best']['validation']['occupied_zone_macro_f1']))
    b.update(primary_view=primary,phase_limit=float(max(np.quantile(cal['quality'][:,0],.995)+.05,.1)),coherence_limit=float(max(0.,np.quantile(cal['quality'][:,1],.005)-.05)))
    report={}
    for view,e in b['experts'].items():
        pred,post,_=expert_prob(b,view,cal[view]);logs=np.log(np.maximum(post,1e-20))
        def nll(t):return float(-np.log(np.maximum(posterior(post,t)[np.arange(len(y)),tid],1e-20)).mean())
        t=float(minimize_scalar(nll,bounds=(.5,4),method='bounded').x);e['temperature']=t;e['roughness_limit']=float(max(np.quantile(cal['quality'][:,2 if view=='calibrated' else 3],.995)*1.5,.08))
        report[view]=dict(temperature=t,nll_before=nll(1),nll_after=nll(t),roughness_limit=e['roughness_limit'])

    b['layout_gate']=.95;r=batch(b,cal);correct=np.all(r['pred']==y,axis=1);confidence=r['posterior'].max(1);eligible=[]
    for gate in np.linspace(.35,.99,65):
        keep=r['quality']&(confidence>=gate)
        if keep.sum()>=40 and correct[keep].mean()>=.9:eligible.append(float(gate))
    b['layout_gate']=min(eligible) if eligible else .99
    b['calibration_note']='600 new calibration scenes only. 90% empirical accepted-layout target with >=40 records is not a guarantee. Unknown is a calibrated quality guard, not universal OOD detection.'
    r=batch(b,cal);b['calibration_report']=dict(experts=report,primary_view=primary,phase_limit=b['phase_limit'],coherence_limit=b['coherence_limit'],layout_gate=b['layout_gate'],gate_target_met=bool(eligible),metrics=enrich(y,r))

    needs_old=False
    for e in b['experts'].values():
        cfg=e['decoder'];w=cfg['tree_weight']
        if w==1 or cfg['net_source']=='v2':e['components'].pop('occ',None)
        if not w:e['components'].pop('tree',None)
        needs_old|=w<1 and cfg['net_source']!='new'
    if not needs_old:b.pop('old_occupancy',None)
    path=NIGHT/'model.joblib';safe_dump(b,path)

    frozen=dict(frozen_at=now(),model_path=str(path),model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),inference_source_sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ['night_infer.py','night_features.py','night_phase.py','opt_v2_common.py']},primary_view=primary,expert_decoders={v:e['decoder'] for v,e in b['experts'].items()},calibration=b['calibration_report'],source_selection_sha256=hashlib.sha256((NIGHT/'selection.json').read_bytes()).hexdigest(),test_opened=False,release_rule='On both new validation and test: improve all-input raw exact, nonworse Macro-F1, each 2/3-count exact nonworse, all-input quality-adjusted operational exact nonworse. No test-based tuning.')
    atomic('candidate_frozen.json',frozen);log('Candidate frozen; new final test may now be opened.');return b

def paired_interval(y,old,new):
    rng=np.random.default_rng(2026103007);delta=np.all(y==new,axis=1).astype(float)-np.all(y==old,axis=1)

    strata=[np.flatnonzero(y.sum(1)==n) for n in [1,2,3]];values=[]
    for _ in range(2000):
        ix=np.concatenate([rng.choice(a,len(a),replace=True) for a in strata if len(a)]);values.append(float(delta[ix].mean()))
    return dict(exact_layout_difference=float(delta.mean()),paired_bootstrap_95_interval=np.quantile(values,[.025,.975]).tolist(),replicates=2000,seed=2026103007)

def phase_diagnostics(b,old):
    jobs=jobs_for('test')[:90];Y=[];base=[];conditions={k:[] for k in ['normal','per_link_phase','per_frame_link_phase','common_phase','opposite_frame_phase']};rng=np.random.default_rng(2026103008)
    for folder,row in jobs:
        with np.load(folder/'records'/(row['id']+'.npz')) as a:h=window(a['h'],rng,30)
        Y.append(np.isin(np.arange(9),[o['zone'] for o in row['objects']]).astype(int));base.append(h)
        conditions['normal'].append(h);conditions['per_link_phase'].append(h*np.exp(1j*rng.uniform(-np.pi,np.pi,(1,32,1))))
        conditions['per_frame_link_phase'].append(h*np.exp(1j*rng.uniform(-np.pi,np.pi,(16,32,1))))
        conditions['common_phase'].append(h*np.exp(1j*1.0));h2=np.repeat(h[:1],16,axis=0);h2[8:]*=-1;conditions['opposite_frame_phase'].append(h2)
    y=np.array(Y);report={};normal_free=None
    for condition,signals in conditions.items():


        valid=[];vv=[];invalid=[]
        for i,h in enumerate(signals):
            try:vv.append(views(h,b['reference']));valid.append(i)
            except ValueError as ex:invalid.append(dict(index=i,reason=str(ex)))
        ix=np.array(valid,dtype=int);n=len(y)
        r=dict(pred=np.zeros((n,9),bool),posterior=np.zeros((n,len(MASKS))),quality=np.zeros(n,bool),accepted=np.zeros(n,bool),routing=np.full(n,'quality_rejected',dtype=object))
        br=dict(pred=np.zeros((n,9),bool),posterior=np.zeros((n,len(MASKS))),quality=np.zeros(n,bool),accepted=np.zeros(n,bool))
        if valid:
            d={k:np.array([v[k] for v in vv]) for k in ['calibrated','phase_free']};d['quality']=np.array([[v[k] for k in QUALITY] for v in vv]);vr=batch(b,d);vbr=baseline(old,d)
            for k in r:r[k][ix]=vr[k]
            for k in br:br[k][ix]=vbr[k]
        if invalid:
            from opt_v2_infer import infer_v2
            for bad in invalid:
                i=bad['index'];actual=infer_v3(b,signals[i]);assert actual['status']=='Unknown'
                legacy=infer_v2(old,signals[i]);br['quality'][i]=legacy['status']!='Unknown';br['accepted'][i]=legacy['status']=='Accepted regions'
                for region in legacy['regions']:br['pred'][i,region['zone']]=1
        m=enrich(y,r);m['phase_free_routed']=int(np.sum(r['routing']=='phase_free'));m['unusable_reference_records']=invalid
        if 'phase_free' in b['experts']:
            free_pred=np.zeros((n,9),bool)
            if valid:free_pred[ix]=expert_prob(b,'phase_free',d['phase_free'])[0]
            if condition=='normal':normal_free=free_pred
            m['free_expert_same_predictions_as_normal']=float(np.all(free_pred==normal_free,axis=1).mean()) if condition!='opposite_frame_phase' else None

        m['v2_raw']=enrich(y,br);report[condition]=m
    report['scope']='90 newly sealed test geometries, synthetic phase perturbations; not real hardware generalization. Opposite-frame case repeats one frame and tests cancellation handling only.'
    return report

def main():
    with stage_lock('evaluation'):
        atomic('evaluation_state.json',dict(state='waiting_for_selection',pid=os.getpid(),updated=now()))
        with threadpool_limits(limits=4):
            frozen=NIGHT/'candidate_frozen.json'
            if frozen.exists():
                fr=json.loads(frozen.read_text(encoding='utf-8'));path=NIGHT/'model.joblib';assert hashlib.sha256(path.read_bytes()).hexdigest()==fr['model_sha256'];b=joblib.load(path)
            else:b=calibrate()
            old=joblib.load(OUT2/'model.joblib');report=dict(version=3,selected_model=b['name'],measurement='16 distinct noisy receiver frames; joint fixed-room Sionna RT; calibrated empty-room reference.',selection='Training and independent new validation for model/decoder choice; independent calibration for uncertainty. Final test opened only after candidate_frozen.json.',training=dict(physical_scenes=36000,new_scenes=21000,original_training_only=15000,observations=72000,augmentation=2,features=1600,phase_free_features=1088),split_counts=dict(train=36000,validation=1200,calibration=600,test=1500,boundary_test=450),comparison=[],selected_decoder=b['experts'][b['primary_view']]['decoder'],splits={},baseline_comparison={},paired_comparison={},primary_view=b['primary_view'],calibration=b['calibration_report'],model_bytes=(NIGHT/'model.joblib').stat().st_size)
            for split in ['validation','calibration','test','boundary_test']:
                if split in ['test','boundary_test']:
                    if not (NIGHT/'test_opened.json').exists():atomic('test_opened.json',dict(first_opened_at=now(),frozen_model_sha256=json.loads((NIGHT/'candidate_frozen.json').read_text(encoding='utf-8'))['model_sha256'],note='Final scoring begins after model and calibration freeze. No retuning after this event.'))
                    build(split)
                d=load(split);r=batch(b,d);br=baseline(old,d);report['splits'][split]=enrich(d['labels'],r);report['baseline_comparison'][split]={'v2_same_frames':enrich(d['labels'],br)};report['paired_comparison'][split]=paired_interval(d['labels'],br['pred'],r['pred'])
                prediction_path=NIGHT/('predictions_'+split+'.npz')
                if prediction_path.exists():
                    with np.load(prediction_path) as previous:
                        assert np.array_equal(previous['pred'],r['pred']) and np.array_equal(previous['accepted'],r['accepted']) and np.array_equal(previous['quality'],r['quality']) and np.allclose(previous['posterior'],r['posterior'],rtol=0,atol=1e-12),'Frozen predictions changed during report resumption'
                else:np.savez_compressed(prediction_path,ids=d['ids'],truth=d['labels'],pred=r['pred'],posterior=r['posterior'],accepted=r['accepted'],quality=r['quality'],routing=r['routing'].astype(str),v2_pred=br['pred'])
                atomic('evaluation_state.json',dict(state='evaluating',pid=os.getpid(),updated=now(),split=split,metrics=report['splits'][split]));log('v3 '+split+' exact='+str(report['splits'][split]['exact_occupied_layout_accuracy']))
            report['phase_diagnostics']=phase_diagnostics(b,old)

            folder,row=jobs_for('test')[0]
            with np.load(folder/'records'/(row['id']+'.npz')) as a:h=window(a['h'],np.random.default_rng(7),30)
            timings=[]
            for i in range(35):
                st=time.perf_counter();infer_v3(b,h)
                if i>=5:timings.append((time.perf_counter()-st)*1000)
            report['inference_ms']=dict(median=float(np.median(timings)),p95=float(np.percentile(timings,95)),scope='Signal features + classifiers + confidence; excludes RT, HTTP and acquisition')


            checks={}
            for split in ['validation','test']:
                n=report['splits'][split];o=report['baseline_comparison'][split]['v2_same_frames']
                checks[split]=dict(exact_improves=n['exact_occupied_layout_accuracy']>o['exact_occupied_layout_accuracy'],macro_f1_nonworse=n['occupied_zone_macro_f1']>=o['occupied_zone_macro_f1'],each_two_three_nonworse=all(n['by_true_count'][str(k)]['exact_occupied_layout_accuracy']>=o['by_true_count'][str(k)]['exact_occupied_layout_accuracy'] for k in [2,3]),operational_exact_nonworse=n['all_inputs_operational_exact_accuracy']>=o['all_inputs_operational_exact_accuracy'])
            report['release_check']=dict(eligible=all(all(a.values()) for a in checks.values()),checks=checks,note='Final test is a one-time release check, not used to tune the model.')
            atomic('results.json',report);atomic('evaluation_state.json',dict(state='completed',pid=os.getpid(),updated=now(),release=report['release_check']));log('Final evaluation complete. Release eligible='+str(report['release_check']['eligible']))

if __name__=='__main__':
    try:main()
    except Exception as e:
        atomic('evaluation_state.json',dict(state='failed',pid=os.getpid(),updated=now(),error=str(e),traceback=traceback.format_exc()));raise
