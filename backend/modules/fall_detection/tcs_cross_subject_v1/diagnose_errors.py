"""Frozen D-model diagnostics: raw integrity, cache equality, crop sensitivity.

Only train/validation metadata and subject03 raw files are inspected. Shifted SA
windows are a declared sensitivity probe, never replacement validation metrics.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
from scipy import signal
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
MODULE=HERE.parent
sys.path.insert(0,str(HERE))
from prepare_features import audited_bytes,decode_amplitude,make_feature,sha
from train_comparison import CNN,protect
from dataset import GCVDataset
sys.path.insert(0,str(MODULE/'tcs_fall_audit_v1'))
from audit_local_archives import fields,reference_check

ARMS=['SA_raw_GCV','SA_CT_raw_GCV','SA_standardized_GCV','SA_CT_standardized_GCV']

def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False))

def records_csv(path):
    with path.open() as f:return list(csv.DictReader(f))

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();out=args.output
    if out.exists():raise FileExistsError('Use a fresh diagnosis directory')
    features=MODULE/'outputs/tcs_cross_subject_v1/features_v1'
    training=MODULE/'reports/tcs_cross_subject_v1/comparison_seed42_v1'
    evaluation=training/'evaluation'
    modelroot=MODULE/'outputs/tcs_cross_subject_v1/comparison_seed42_v1'
    eligibility=MODULE/'reports/tcs_cross_subject_v1/eligibility_20261029'
    sources=[modelroot/a/'best.pt' for a in ARMS]+[MODULE/'tcs_withinsubject_v1/model.py',MODULE/'tcs_withinsubject_v1/prepare.py',HERE/'train_comparison.py']
    protected={str(p):sha(p) for p in sources};legacy=protect()
    prep=json.loads((features/'preparation.json').read_text());samples=json.loads((features/'samples.json').read_text())
    assert sha(features/'features.f32')==prep['feature_cache_sha256']
    assert sha(features/'samples.json')==prep['sample_manifest_sha256']
    rows={r['sample_id']:r for r in samples};cache=np.memmap(features/'features.f32',dtype=np.float32,mode='r',shape=tuple(prep['shape']))
    audit={r['archive_member']:r for r in json.loads((eligibility/'records.json').read_text())}
    eligibility_info=json.loads((eligibility/'eligibility.json').read_text())
    archive=next(x for x in eligibility_info['archives'] if x['subject']==3)
    assert sha(Path(archive['path']))==archive['sha256']
    predictions={a:{r['sample_id']:r for r in records_csv(training/a/'best_validation_predictions.csv')} for a in ARMS}
    d=predictions[ARMS[3]]
    fn=[id for id,r in d.items() if r['kind']=='single_action' and r['label']=='1' and r['prediction']=='0']
    fp=[id for id,r in d.items() if r['kind']=='single_action' and r['label']=='0' and r['prediction']=='1']
    assert len(fn)==6 and len(fp)==3
    assert {rows[id]['activity'] for id in fn}=={2}
    selected=sorted({r['source_member'] for r in samples if r['role']=='validation' and r['kind']=='single_action' and r['activity']==2}|{rows[id]['source_member'] for id in fp})
    ct_members=['3/a_03_13_00.dat','3/a_03_13_01.dat']
    assert all(audit[m]['role']=='validation' and audit[m]['nominal_subject_id']==3 for m in selected+ct_members)
    out.mkdir(parents=True)
    write(out/'diagnostic_protocol.json',{'status':'specified before new diagnostic predictions','frozen_threshold':.5,
        'main_metrics':'original first8000-packet validation and original continuous replay unchanged',
        'raw_subjects':[3],'test_evaluated':False,'retraining':False,
        'SA_raw_selection':'all20 validation activity2 records and all3 D nonfall mistakes',
        'crop_sensitivity':'all available 8000-packet windows at start0,1000,... plus final complete window; scores only, no new accuracy or replacement checkpoint selection',
        'trace':'mean over90 features of raw-amplitude CV in disjoint20-packet blocks across full recording; diagnostic proxy, not model GCV or fall annotation',
        'inference_backend':'cpu float32 eval; numerical comparison to saved MPS scores with tolerance5e-5',
        'protected_files_before':protected,'legacy_before':legacy})
    state=torch.load(modelroot/ARMS[3]/'best.pt',map_location='cpu',weights_only=True)
    assert state['threshold']==.5 and state['label_mapping']=={'Nonfall':0,'Fall':1}
    norm=state['normalization_parameters'];savednorm=json.loads((features/'normalization_sa_ct.json').read_text())
    assert norm==savednorm and norm['fit_subjects']==[1,2] and norm['fit_role']=='train'
    mean=np.array(norm['mean'],np.float32);scale=np.array(norm['scale'],np.float32)
    assert (scale>0).all() and mean.shape==scale.shape==(90,)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    model=CNN().eval();model.load_state_dict(state['model_state_dict'],strict=True)
    assert not any(m.training for m in model.modules())
    def score(x):
        x=(np.asarray(x,np.float32)-mean)/scale
        if x.ndim==3:x=x[None]
        assert x.shape[1:]==(1,400,90) and np.isfinite(x).all()
        logits=model(torch.from_numpy(x.copy()));assert torch.isfinite(logits).all()
        return logits.softmax(1)[:,1].numpy()
    val=[r for r in samples if r['role']=='validation' and r['kind']=='single_action']
    cpu_scores=[]
    for i in range(0,len(val),16):cpu_scores.extend(score(cache[[r['cache_index'] for r in val[i:i+16]]]).tolist())
    old=np.array([float(d[r['sample_id']]['fall_score']) for r in val]);new=np.array(cpu_scores)
    error=float(np.max(np.abs(old-new)));flips=int(((old>=.5)!=(new>=.5)).sum())
    assert error<5e-5 and flips==0
    dataset=GCVDataset(features,'validation',True,features/'normalization_sa_ct.json')
    dataset_lookup={samples[i]['sample_id']:j for j,i in enumerate(dataset.indices)}
    for id in fn+fp:
        x,y,sid=dataset[dataset_lookup[id]];expected=(cache[rows[id]['cache_index']]-mean)/scale
        assert np.array_equal(x.numpy(),expected) and int(y)==int(d[id]['label']) and sid==id
    write(out/'runtime_checks.json',{'D_checkpoint_normalization_matches_saved_train_statistics':True,
        'mean_scale_shapes':[list(mean.shape),list(scale.shape)],'model_eval_all_layers':True,
        'CPU_vs_saved_MPS_samples':len(val),'max_abs_score_difference':error,'class_changes':flips,
        'dataset_manual_normalization_exact_for9_SA_errors':True,'padding':False})
    b,a=signal.butter(8,.2,'lowpass');details=[];crop_rows=[];traces={};feature_rows=[]
    official=MODULE/'outputs/tcs_fall_audit_v1/upstream/code/wifilib.py'
    def inspect_native(member,data,amp,t):
        frames=np.frombuffer(data,np.uint8).reshape(-1,575);ts,counts,permutation=fields(frames)
        chosen=sorted(set([0,min(7999,len(frames)-1),len(frames)-1]+[int(np.flatnonzero(frames[:,18]==v)[0]) for v in np.unique(frames[:,18])]))
        reference=reference_check(official,frames,permutation,chosen)
        delta=np.diff(t);jumps=(np.diff(counts)%65536)
        return {'member':member,'role':'validation','activity':audit[member]['activity_id'],
            'CRC_and_raw_SHA_match':True,'packet_count':len(amp),'duration_seconds':float(t[-1]),
            'native_selected_shape':list(amp.shape),'finite_ratio':float(np.isfinite(amp).mean()),
            'nonincreasing_timestamps':int((delta<=0).sum()),'max_packet_gap_seconds':float(delta.max()),
            'counter_steps_not1':int((jumps!=1).sum()),'reference':reference,
            'recorded_annotation':audit[member]['annotation_record']}
    with zipfile.ZipFile(archive['path']) as z:
        for n,member in enumerate(selected,1):
            rec=audit[member];data=audited_bytes(z,rec);amp,t=decode_amplitude(data)
            detail=inspect_native(member,data,amp,t)
            base_feature=make_feature(amp[:8000],b,a);id=member+':packet0';cached=cache[rows[id]['cache_index']]
            diff=float(np.max(np.abs(base_feature-cached)));assert diff==0
            starts=sorted(set(range(0,len(amp)-8000+1,1000))|{len(amp)-8000})
            feats=np.stack([make_feature(amp[start:start+8000],b,a) for start in starts]);scores=score(feats)
            detail.update({'first_window_cache_max_abs_difference':diff,'first_window_end_seconds':float(t[7999]),
                'omitted_tail_packets':len(amp)-8000,'omitted_tail_seconds':float(t[-1]-t[7999]),
                'label':int(d[id]['label']),'original_score':float(d[id]['fall_score']),'was_D_FN':id in fn,'was_D_FP':id in fp})
            for start,s in zip(starts,scores):crop_rows.append({'member':member,'activity':rec['activity_id'],'was_D_FN':id in fn,
                'start_packet':start,'start_seconds':float(t[start]),'end_seconds':float(t[start+7999]),'fall_score':float(s),'prediction':int(s>=.5)})
            # Full-record visualization uses no class labels or fitted parameters.
            count=len(amp)//20;g=amp[:count*20].astype(np.float64).reshape(count,20,90);mu=g.mean(1)
            rawcv=np.divide(g.std(1),mu,out=np.zeros_like(mu),where=mu!=0).mean(1)
            time_axis=t[:count*20].reshape(count,20).mean(1);peak=int(rawcv.argmax())
            detail.update({'raw_variability_peak_seconds':float(time_axis[peak]),'raw_variability_peak_outside_first8000':bool(peak>=400),
                'first_window_zero_gcv_values':int((base_feature==0).sum()),'first_window_z_range':[float(((base_feature-mean)/scale).min()),float(((base_feature-mean)/scale).max())]})
            if id in fn:traces[member]=(time_axis,rawcv,float(t[7999]))
            details.append(detail)
            print(json.dumps({'raw_SA_checked':n,'total':len(selected),'member':member,'cache_difference':diff}),flush=True)
            del data,amp,feats,g
        # Replay raw checks for all five positive background windows plus local neighbors and event onset.
        replay=json.loads((evaluation/(ARMS[3]+'.json')).read_text())
        saved_windows=records_csv(evaluation/(ARMS[3]+'_windows.csv'))
        for member in ct_members:
            data=audited_bytes(z,audit[member]);amp,t=decode_amplitude(data);detail=inspect_native(member,data,amp,t)
            windows=[w for w in saved_windows if w['member']==member]
            positive=[i for i,w in enumerate(windows) if float(w['window_end_seconds'])<=120 and w['prediction']=='1']
            onset=next(x['strict_nominal_delay_seconds']+x['annotation_seconds'] for x in replay['continuous']['per_record'] if x['member']==member)
            event_i=min(range(len(windows)),key=lambda i:abs(float(windows[i]['window_end_seconds'])-onset))
            probe=sorted({j for i in positive+[event_i] for j in range(max(0,i-2),min(len(windows),i+3))})
            maxdiff=0.;scorediff=0.
            for i in probe:
                w=windows[i];start=int(w['start_packet']);feature=make_feature(amp[start:start+8000],b,a)
                if w['sample_id'] in rows:
                    dif=float(np.max(np.abs(feature-cache[rows[w['sample_id']]['cache_index']])));assert dif==0;maxdiff=max(maxdiff,dif)
                s=float(score(feature)[0]);scorediff=max(scorediff,abs(s-float(w['fall_score'])))
                assert abs(float(t[start+7999])-float(w['window_end_seconds']))<1e-12
                window_gaps=np.diff(t[start:start+8000])
                feature_rows.append({'member':member,'window_index':int(w['window_index']),'start_packet':start,
                    'start_seconds':float(t[start]),'end_seconds':float(t[start+7999]),'raw_GCV_mean':float(feature.mean()),
                    'raw_GCV_max':float(feature.max()),'z_abs_p99':float(np.quantile(np.abs((feature-mean)/scale),.99)),
                    'max_packet_gap_seconds':float(window_gaps.max()),'CPU_score':s,'saved_MPS_score':float(w['fall_score']),
                    'background':float(w['window_end_seconds'])<=120,'original_prediction':int(w['prediction'])})
            assert scorediff<5e-5
            detail.update({'diagnostic_windows_checked':len(probe),'background_cache_max_abs_difference':maxdiff,
                'raw_CPU_vs_saved_MPS_max_score_difference':scorediff,'original_positive_background_windows':len(positive)})
            details.append(detail);print(json.dumps({'raw_CT_checked':member,'windows':len(probe),'max_score_difference':scorediff}),flush=True)
            del data,amp
    write(out/'raw_checks.json',details)
    for name,rr in [('SA_crop_sensitivity.csv',crop_rows),('CT_local_windows.csv',feature_rows)]:
        with (out/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rr[0]));writer.writeheader();writer.writerows(rr)
    fig,axes=plt.subplots(3,2,figsize=(13,10),constrained_layout=True)
    for ax,(member,(t,cv,end)) in zip(axes.flat,traces.items()):
        ax.plot(t,cv,lw=.8,color='#326f9c');ax.axvspan(0,end,alpha=.10,color='green',label='Original first8000 packets')
        ax.axvline(end,color='black',ls='--',lw=.8);ax.set_title(member);ax.set_xlabel('Raw CSI elapsed seconds')
        ax.set_ylabel('Mean raw-amplitude CV');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Six missed falls: full raw recording | variability is not an event annotation')
    fig.savefig(out/'missed_fall_crop_traces.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    for member in traces:
        rr=[x for x in crop_rows if x['member']==member]
        ax.plot([x['start_seconds'] for x in rr],[x['fall_score'] for x in rr],marker='o',label=member)
    ax.axhline(.5,color='black',ls='--');ax.set_xlabel('Window start time (seconds)');ax.set_ylabel('Frozen D Fall score')
    ax.set_title('Crop sensitivity only: same weights, normalization, threshold; original metrics unchanged')
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.savefig(out/'crop_score_sensitivity.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,7),constrained_layout=True)
    colors=['#3a78a6','#d88c36','#5c9e57','#b34e6f']
    all_windows={a:records_csv(evaluation/(a+'_windows.csv')) for a in ARMS}
    for row,member in enumerate(ct_members):
        for col,(left,right) in enumerate([(20,45),(128,150)]):
            ax=axes[row,col]
            for color,a in zip(colors,ARMS):
                w=[x for x in all_windows[a] if x['member']==member and left<=float(x['window_end_seconds'])<=right]
                ax.plot([float(x['window_end_seconds']) for x in w],[float(x['fall_score']) for x in w],marker='.',label=a,color=color)
            ax.axhline(.5,c='black',ls='--');ax.set_ylim(.15,1.03);ax.set_title(member+(' | background' if col==0 else ' | fall'))
            ax.set_xlabel('Decision endpoint (s)');ax.set_ylabel('Fall score');ax.grid(alpha=.2)
            if col==1:ax.axvline(137,color='red',ls=':')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.get_layout_engine().set(rect=(0,0,1,.94))
    fig.legend(handles,labels,loc='upper center',ncol=2,fontsize=8);fig.savefig(out/'two_false_alarms_and_true_falls.png',dpi=150);plt.close(fig)
    assert protected=={str(p):sha(p) for p in sources}
    summary={'completed':True,'raw_records_rechecked':len(details),'SA_records':len(selected),'CT_records':len(ct_members),
        'raw_subjects':[3],'cache_features_exact':True,'normalization_reused_without_refit':True,
        'runtime_checks':json.loads((out/'runtime_checks.json').read_text()),
        'model_and_source_hashes_unchanged':True,'protected_after':protect(),'test_evaluated':False,'retraining':False,
        'original_validation_metrics_unchanged':True,'source_sha256':sha(Path(__file__))}
    write(out/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
