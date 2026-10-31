"""Fixed-window, frozen-checkpoint record-level robustness experiment.

Uses eligible SA recordings from subjects01/02(train) and03(validation).
Record labels are never assigned to arbitrary individual fall subwindows.
No training, threshold search, test data, or continuous alarm-rule changes.
"""
import argparse
from collections import Counter,defaultdict
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile
import numpy as np
from scipy import signal
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent;MODULE=HERE.parent
sys.path.insert(0,str(HERE))
from prepare_features import audited_bytes,decode_amplitude,make_feature,sha
from train_comparison import CNN,protect,metrics
ARMS=['SA_raw_GCV','SA_CT_raw_GCV','SA_standardized_GCV','SA_CT_standardized_GCV']
POLICIES=['first','center','last','grid_max']

def save(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False))
def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def read_csv(path):
    with path.open() as f:return list(csv.DictReader(f))

def starts_for(n):
    if n<8000:raise ValueError('Short input; no padding')
    grid=sorted(set(range(0,n-8000+1,1000))|{n-8000})
    return sorted(set(grid)|{(n-8000)//2}),grid

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);args=ap.parse_args()
    if args.output.exists() or args.report.exists():raise FileExistsError('Use fresh output and report directories')
    eligibility=MODULE/'reports/tcs_cross_subject_v1/eligibility_20261029'
    original=MODULE/'outputs/tcs_cross_subject_v1/features_v1'
    models=MODULE/'outputs/tcs_cross_subject_v1/comparison_seed42_v1'
    reports=MODULE/'reports/tcs_cross_subject_v1/comparison_seed42_v1'
    info=json.loads((eligibility/'eligibility.json').read_text())
    if sha(eligibility/'records.json')!=info['records_sha256']:raise ValueError('Eligibility records changed')
    cohort=sorted([r for r in json.loads((eligibility/'records.json').read_text()) if r['input_eligibility'].startswith('eligible') and r['role'] in ('train','validation') and r['source_kind']=='single_action'],key=lambda r:r['archive_member'])
    if len(cohort)!=496 or {r['nominal_subject_id'] for r in cohort}!={1,2,3}:raise ValueError('Unexpected cohort')
    prep=json.loads((original/'preparation.json').read_text());original_rows=json.loads((original/'samples.json').read_text())
    lookup={r['source_member']:r for r in original_rows if r['kind']=='single_action'}
    if set(lookup)!={r['archive_member'] for r in cohort}:raise ValueError('Original/new SA membership differs')
    if sha(original/'features.f32')!=prep['feature_cache_sha256'] or sha(original/'samples.json')!=prep['sample_manifest_sha256']:raise ValueError('Original data changed')
    sources=[models/a/'best.pt' for a in ARMS]+[MODULE/'tcs_withinsubject_v1/model.py',MODULE/'tcs_withinsubject_v1/prepare.py',HERE/'train_comparison.py',HERE/'prepare_features.py']
    protected={str(p):sha(p) for p in sources}
    expected=sum(len(starts_for(r['packet_count'])[0]) for r in cohort)
    if shutil.disk_usage(MODULE).free<expected*400*90*4+1024**3:raise RuntimeError('Reserve1GiB beyond new cache')
    args.output.mkdir(parents=True);args.report.mkdir(parents=True)
    protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'frozen before new window predictions',
        'scope':'development record-level SA robustness; motivated by already-seen validation errors; not independent confirmation',
        'subjects':{'train':[1,2],'validation':[3]},'test_subject04_read':False,'threshold':.5,'arms':ARMS,
        'record_count':len(cohort),'record_counts_by_role':dict(Counter(r['role'] for r in cohort)),
        'window_packets':8000,'features':[1,400,90],'expected_unique_windows':expected,
        'policies':{'first':'start0','center':'start floor((complete_packets-8000)/2)','last':'start complete_packets-8000',
                    'grid_max':'max score among starts0,1000,2000,... with complete8000 packets, plus final complete window; includes first and last, excludes center unless on grid'},
        'labels':'only recording-level label; for grid_max any positive window makes recording positive, no claim each subwindow contains a fall',
        'uniformity':'same label-blind rules for every record and arm; no per-record offset choice based on score or label',
        'fairness':'same496 eligible recordings for all policies; grid window count varies with record length, report its distribution',
        'padding':False,'resampling':False,'normalization':'checkpoint training statistics unchanged; never refit',
        'preprocessing':'same window-local filtfilt/GCV; Rx-major90features, Tx0',
        'evaluation':'CPU float32 eval, no dropout; batches16; original first-window validation scores reproduced within5e-5, no class changes',
        'endpoints':'actual NIC timestamp; center,last and offline grid_max use knowledge of recording end; no real-time alert latency claims',
        'no_changes':['training','threshold','old models','original evaluation','continuous alarm rules','test04'],
        'primary_reporting':'D validation record-level TP/FP/FN/TN, Fall precision/recall/F1, nonfall false-positive fraction; other arms and training descriptive',
        'no_selection':'do not choose a production policy or update frozen model from this exploratory comparison',
        'protected_hashes':protected,'legacy_before':protect(),'source_sha256':sha(Path(__file__))}
    save(args.report/'protocol.json',protocol)
    save(args.report/'cohort.json',[{'member':r['archive_member'],'subject':r['nominal_subject_id'],'role':r['role'],'activity':r['activity_id'],'label':r['single_action_binary_label'],'packet_count':r['packet_count'],'raw_sha256':r['raw_file_sha256'],'verified_prefix_bytes':r.get('decoded_complete_prefix_bytes')} for r in cohort])
    original_cache=np.memmap(original/'features.f32',dtype=np.float32,mode='r',shape=tuple(prep['shape']))
    b,a=signal.butter(8,.2,'lowpass');windows=[];record_rows=[];start_time=time.perf_counter()
    with (args.output/'features.f32').open('wb') as f:
        for archive in info['archives']:
            if archive['subject'] not in (1,2,3):continue
            if sha(Path(archive['path']))!=archive['sha256']:raise ValueError('Raw archive changed')
            selected=[r for r in cohort if r['nominal_subject_id']==archive['subject']]
            with zipfile.ZipFile(archive['path']) as z:
                for r in selected:
                    raw=audited_bytes(z,r);amp,t=decode_amplitude(raw);n=len(amp)
                    if n!=r['packet_count']:raise ValueError('Packet count differs from audit')
                    starts,grid=starts_for(n);base=lookup[r['archive_member']];indexmap={}
                    for start in starts:
                        feature=make_feature(amp[start:start+8000],b,a)
                        if start==0 and not np.array_equal(feature,original_cache[base['cache_index']]):raise ValueError('First-window cache mismatch')
                        if feature.shape!=(1,400,90) or not np.isfinite(feature).all():raise ValueError('Invalid features')
                        idx=len(windows);indexmap[start]=idx;f.write(np.ascontiguousarray(feature).tobytes())
                        windows.append({'cache_index':idx,'member':r['archive_member'],'subject':r['nominal_subject_id'],'role':r['role'],'activity':r['activity_id'],
                            'start_packet':start,'end_packet':start+7999,'start_seconds':float(t[start]),'end_seconds':float(t[start+7999]),'on_grid':start in grid})
                    record_rows.append({'member':r['archive_member'],'subject':r['nominal_subject_id'],'role':r['role'],'activity':r['activity_id'],'label':r['single_action_binary_label'],
                        'packets':n,'duration_seconds':float(t[-1]),'grid_window_count':len(grid),'unique_window_count':len(starts),
                        'policy_indices':{'first':[indexmap[0]],'center':[indexmap[(n-8000)//2]],'last':[indexmap[n-8000]],'grid_max':[indexmap[x] for x in grid]}})
                    if len(record_rows)%25==0 or len(selected)==sum(v['subject']==r['nominal_subject_id'] for v in record_rows):
                        progress={'stage':'raw_preparation','records':len(record_rows),'total':len(cohort),'windows':len(windows),'elapsed_seconds':time.perf_counter()-start_time}
                        save(args.report/'progress.json',progress);print(json.dumps(progress),flush=True)
                    del amp,raw
    shape=(len(windows),1,400,90)
    if len(windows)!=expected or (args.output/'features.f32').stat().st_size!=int(np.prod(shape))*4:raise ValueError('Unexpected window cache size')
    save(args.output/'windows.json',windows);save(args.output/'record_indices.json',record_rows)
    feature_hash=sha(args.output/'features.f32');save(args.output/'preparation.json',{'shape':shape,'feature_cache_sha256':feature_hash,'windows_sha256':sha(args.output/'windows.json'),'record_indices_sha256':sha(args.output/'record_indices.json'),'all496_original_first_features_exact':True,'test_read':False})
    cache=np.memmap(args.output/'features.f32',dtype=np.float32,mode='r',shape=shape)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    summary=[];group_metrics=[];record_scores=[];stability=[];reload=[];all_scores={}
    for arm in ARMS:
        state=torch.load(models/arm/'best.pt',map_location='cpu',weights_only=True)
        if state['threshold']!=.5 or state['label_mapping']!={'Nonfall':0,'Fall':1}:raise ValueError('Checkpoint protocol mismatch')
        model=CNN().eval();model.load_state_dict(state['model_state_dict'],strict=True)
        norm=state['normalization_parameters'];mean=scale=None
        if norm is not None:
            if norm['fit_role']!='train' or norm['fit_subjects']!=[1,2] or norm['feature_cache_sha256']!=prep['feature_cache_sha256']:raise ValueError('Normalizer provenance mismatch')
            mean=np.array(norm['mean'],np.float32);scale=np.array(norm['scale'],np.float32)
        scores=[]
        for start in range(0,len(cache),16):
            x=np.array(cache[start:start+16],copy=True)
            if mean is not None:x=(x-mean)/scale
            if not np.isfinite(x).all():raise ValueError('Invalid normalized feature')
            logits=model(torch.from_numpy(x));assert torch.isfinite(logits).all()
            scores.extend(logits.softmax(1)[:,1].tolist())
        scores=np.array(scores);all_scores[arm]=scores
        old={r['sample_id']:r for r in read_csv(reports/arm/'best_validation_predictions.csv')}
        diffs=[];flips=0
        for r in record_rows:
            vals={p:float(scores[ix].max()) for p,ix in r['policy_indices'].items()}
            common={k:r[k] for k in ('member','subject','role','activity','label','grid_window_count','unique_window_count')}
            for policy,value in vals.items():record_scores.append({**common,'arm':arm,'policy':policy,'fall_score':value,'prediction':int(value>=.5)})
            v=scores[sorted(set(j for ix in r['policy_indices'].values() for j in ix))]
            stability.append({**common,'arm':arm,'min_score':float(v.min()),'max_score':float(v.max()),'score_range':float(np.ptp(v)),
                              'any_judgment_flip':bool(v.min()<.5<=v.max()),'first_prediction':int(vals['first']>=.5)})
            if r['role']=='validation':
                before=float(old[r['member']+':packet0']['fall_score']);diffs.append(abs(before-vals['first']));flips+=int((before>=.5)!=(vals['first']>=.5))
        if max(diffs)>5e-5 or flips:raise ValueError('Frozen first-window baseline not reproduced')
        reload.append({'arm':arm,'validation_samples':len(diffs),'max_abs_score_difference':max(diffs),'class_changes':flips})
        for policy in POLICIES:
            for role in ('train','validation'):
                group=[r for r in record_scores if r['arm']==arm and r['policy']==policy and r['role']==role]
                m=metrics([r['label'] for r in group],[r['fall_score'] for r in group]);m['nonfall_false_positive_rate']=m['fp']/m['nonfall']
                summary.append({'arm':arm,'policy':policy,'role':role,**m})
                for subject in sorted({r['subject'] for r in group}):
                    for activity in range(7):
                        sub=[r for r in group if r['subject']==subject and r['activity']==activity]
                        if not sub:continue
                        group_metrics.append({'arm':arm,'policy':policy,'role':role,'subject':subject,'activity':activity,'n':len(sub),'fall':sum(r['label'] for r in sub),
                            'positive_predictions':sum(r['prediction'] for r in sub),'errors':sum(r['prediction']!=r['label'] for r in sub)})
        write_csv(args.report/(arm+'_window_scores.csv'),[{**r,'fall_score':float(s),'prediction':int(s>=.5)} for r,s in zip(windows,scores)])
        print(json.dumps({'arm_evaluated':arm,'windows':len(scores),'baseline_reload':reload[-1],
                          'validation':[v for v in summary if v['arm']==arm and v['role']=='validation']}),flush=True)
        save(args.report/'progress.json',{'stage':'model_evaluation','completed_arm':arm,'windows':len(scores)})
        del model,state
    write_csv(args.report/'record_scores.csv',record_scores);write_csv(args.report/'record_stability.csv',stability)
    write_csv(args.report/'group_metrics.csv',group_metrics);save(args.report/'record_metrics.json',summary);save(args.report/'reload_checks.json',reload)
    transitions=[]
    for arm in ARMS:
        rr={}
        for x in record_scores:
            if x['arm']==arm and x['role']=='validation':rr.setdefault(x['member'],{})[x['policy']]=x
        for policy in POLICIES[1:]:
            for member,by_policy in rr.items():
                base,new=by_policy['first'],by_policy[policy]
                if base['prediction']!=new['prediction']:
                    transitions.append({'arm':arm,'policy':policy,'member':member,'activity':base['activity'],'label':base['label'],
                        'first_score':base['fall_score'],'new_score':new['fall_score'],'first_prediction':base['prediction'],'new_prediction':new['prediction'],
                        'change':'corrected' if new['prediction']==new['label'] else 'new_error'})
    write_csv(args.report/'validation_transitions.csv',transitions)
    # Training is descriptive; validation is the development comparison.
    fig,axes=plt.subplots(1,3,figsize=(14,4.6),constrained_layout=True);x=np.arange(4)
    for arm,label in zip(ARMS,['A','B','C','D']):
        rr=[next(v for v in summary if v['arm']==arm and v['role']=='validation' and v['policy']==p) for p in POLICIES]
        for ax,field in zip(axes,['recall','nonfall_false_positive_rate','f1']):ax.plot(x,[100*r[field] for r in rr],marker='o',label=label)
    for ax,title in zip(axes,['Fall recall (%)','Nonfall record false-positive rate (%)','Fall F1 (%)']):
        ax.set_title(title);ax.set_xticks(x,['First','Center','Last','Grid max']);ax.set_xlabel('Fixed record policy');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Validation subject03 | frozen checkpoints, threshold0.5 | exploratory')
    fig.savefig(args.report/'validation_policy_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for ax,role in zip(axes,['train','validation']):
        for label in (0,1):
            vals=[r['score_range'] for r in stability if r['arm']==ARMS[3] and r['role']==role and r['label']==label]
            ax.boxplot(vals,positions=[label],widths=.5,showmeans=True)
        ax.set_xticks([0,1],['Nonfall','Fall']);ax.set_ylabel('Max score - min score across all probes');ax.set_title('D | '+role);ax.grid(axis='y',alpha=.2)
    fig.savefig(args.report/'D_score_variation.png',dpi=160);plt.close(fig)
    result={'status':'completed','protocol_sha256':sha(args.report/'protocol.json'),'feature_cache_sha256':feature_hash,
        'records':len(record_rows),'unique_windows':len(windows),'raw_subjects':[1,2,3],
        'grid_window_count_distribution':dict(sorted(Counter(r['grid_window_count'] for r in record_rows).items())),
        'record_metrics':summary,'baseline_reload_checks':reload,'original_first_features_exact':True,
        'normalization_refit':False,'threshold_search':False,'training':False,'test_evaluated':False,
        'protected_after':protect(),'original_evaluation_preserved':True,'source_sha256':sha(Path(__file__))}
    if protected!={str(p):sha(p) for p in sources}:raise ValueError('Protected files changed')
    save(args.report/'results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('record_metrics','protected_after')},indent=2),flush=True)

if __name__=='__main__':main()
