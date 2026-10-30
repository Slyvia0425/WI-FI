"""Audit original CSI and assess four frozen isolated-spike preprocessing variants."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.io import loadmat,whosmat

from ..preprocessing.isolated_spikes import isolated_spike_filter,resample_amplitudes,preprocessing_configs
from ..preprocessing.unified_rate import ESTIMATOR_ID,estimate_unified_rate
from ..preprocessing.wire_breath import WireBreathDataset
from ..preprocessing.wire_cache import load_wire_breath_cache
from ..preprocessing.wire_protocol import DEFAULT_PATH,load_protocol
from .evaluate_local_rate_peaks import evaluate

ROOT=Path('backend/modules/breathing_detection/outputs')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def excursion_flags(values):
    values=np.asarray(values,dtype=float)
    median=np.median(values,axis=0)
    scale=1.4826*np.maximum(np.median(abs(values-median),axis=0),1e-8)
    return np.mean(abs(values-median)>6*scale,axis=1)>=.5


def raw_slice(time,values,targets):
    lo=max(0,int(np.searchsorted(time,targets[0],side='right'))-1)
    hi=min(len(time),int(np.searchsorted(time,targets[-1],side='left'))+1)
    return time[lo:hi],values[lo:hi]


def candidate_window(time,values,targets,current,config):
    if config['placement']=='raw':
        clipped_t,clipped_x=raw_slice(time,values,targets)
        cleaned,edits=isolated_spike_filter(clipped_t,clipped_x,config['threshold'])
        x=resample_amplitudes(clipped_t,cleaned,targets)
    elif config['placement']=='resampled':
        cleaned,edits=isolated_spike_filter(targets,current[0].T,config['threshold'])
        x=np.ascontiguousarray(cleaned.T[None],dtype=np.float32)
    else:
        raise ValueError('Unknown preprocessing placement')
    return x,edits


def audit_raw(time,values,complex_values,targets,current):
    reconstructed=resample_amplitudes(time,values,targets)
    clipped_t,clipped_x=raw_slice(time,values,targets)
    flags=excursion_flags(clipped_x)
    dt=np.diff(clipped_t)
    preceding_flags=flags[1:]
    means=np.mean(values,axis=1)
    peak=int(np.argmax(means))
    peak_vector=np.asarray(values[peak],dtype=np.float32)
    complex_error=abs(np.abs(complex_values)-values) if complex_values is not None else None
    return {'raw_shape':list(values.shape),'complex_available':complex_values is not None,
            'complex_abs_max_error':float(np.max(complex_error)) if complex_error is not None else None,
            'complex_abs_close':bool(np.allclose(np.abs(complex_values),values,rtol=1e-6,atol=1e-5)) if complex_values is not None else None,
            'time_start':float(time[0]),'time_end':float(time[-1]),'window_start':float(targets[0]),
            'start_offset_from_first_csi':float(targets[0]-time[0]),
            'time_strictly_increasing':bool(np.all(np.diff(time)>0)),
            'all_values_finite':bool(np.isfinite(values).all()),
            'raw_frame_mean_quantiles':np.quantile(means,[.5,.99,1]).tolist(),
            'raw_window_frame_mean_max':float(clipped_x.mean(axis=1).max()),
            'resampled_frame_mean_max':float(current[0].mean(axis=0).max()),
            'adjacent_exact_duplicate_frames':int(np.sum(np.all(values[1:]==values[:-1],axis=1))),
            'peak_frame_hash':hashlib.sha256(peak_vector.tobytes()).hexdigest(),
            'peak_frame_occurrences_in_file':int(np.sum(np.all(np.asarray(values,dtype=np.float32)==peak_vector,axis=1))),
            'raw_window_points':len(clipped_t),'raw_excursion_frames':int(flags.sum()),
            'resampled_excursion_frames':int(excursion_flags(current[0].T).sum()),
            'window_interval_median':float(np.median(dt)),'window_interval_max':float(dt.max()),
            'interval_before_excursion_median':float(np.median(dt[preceding_flags])) if preceding_flags.any() else None,
            'interval_before_nonexcursion_median':float(np.median(dt[~preceding_flags])) if (~preceding_flags).any() else None,
            'resampling_matches_loader':bool(np.array_equal(reconstructed,current)),
            'resampling_max_abs_difference':float(np.max(abs(reconstructed-current)))}


def select_preprocessing(rows,methods):
    candidates={method:evaluate(rows,method) for method in methods}
    eligible=[method for method,m in candidates.items() if m['retained']==len(rows) and m['good_harm_rate']==0]
    changed=lambda method:sum(r['predictions'][method]!=r['predictions']['baseline'] for r in rows)
    chosen=min(eligible,key=lambda method:(candidates[method]['record_macro_mae_bpm'],changed(method),methods.index(method)))
    return chosen,candidates


def main():
    output=ROOT/'spike_preprocessing_v1'
    if output.exists():
        raise FileExistsError('Preserve configured spike experiment; use a reviewed new version')
    source=ROOT/'respiratory_rate_unified_v1/window_predictions.json'
    originals=json.loads(source.read_text())
    by_id={r['sample_id']:r for r in originals}
    if len(by_id)!=len(originals) or any(r['estimate']['estimator_id']!=ESTIMATOR_ID for r in originals):
        raise ValueError('Duplicate or non-unified baseline source')
    cache=load_wire_breath_cache(ROOT/'wire_breath_cache',1401)
    protocol=load_protocol(DEFAULT_PATH,cache.manifest['samples'])
    configs=preprocessing_configs()
    methods=['baseline']+[c['name'] for c in configs]
    protected=[source,ROOT/'respiratory_rate_unified_v1/metrics.json',ROOT/'lstm_interpolated_full_seed123/best_model.pt',
               ROOT/'replay.json',Path('frontend/src/modules/breathing_detection/data/replay.json'),Path('frontend/package-lock.json')]
    config={'baseline_estimator_id':ESTIMATOR_ID,'protocol_sha256':protocol['sample_fingerprint_sha256'],
            'preprocessing_candidates':configs,'filter_spec':{'half_window_seconds':.1,'min_local_points':5,
                'mad_scale':1.4826,'global_mad_floor_fraction':.05,'absolute_mad_floor':1e-8,
                'max_run_points':2,'max_bracket_seconds':.15,'max_original_interval_seconds':.1},
            'selection':'minimum inner-validation record-macro MAE among baseline/candidates with full coverage and zero current-good harm; ties fewer rate changes then config order',
            'support':'zero harm, full coverage, >=25% reduction in >5 BPM errors and no person MAE worsening; no automatic application change',
            'implementation_sha256':digest(Path(__file__).parents[1]/'preprocessing/isolated_spikes.py'),
            'evaluation_sha256':digest(__file__),'protected_artifact_hashes':{str(p):digest(p) for p in protected},
            'application_enabled':False,'limitations':['Retrospective development data; not new independent validation.',
                'Filter statistics are CSI-only within each window; reference is unchanged and used for evaluation only.',
                'No physical cause inferred from excursions, repetition or time association.',
                'Fold-selected preprocessing is experimental, not a runtime person-specific algorithm.']}
    output.mkdir()
    (output/'config.json').write_text(json.dumps(config,indent=2))
    dataset=WireBreathDataset('data/WiReBreath',window_seconds=30,stride_seconds=30,normalize=False)
    rows,audits=[],[]
    for i,ref in enumerate(dataset._windows):
        sample=dataset[i]
        raw=loadmat(ref.path,squeeze_me=True,variable_names=('time','csi_abs','csi_complex'))
        t=np.asarray(raw['time'],dtype=float).reshape(-1)
        values=np.asarray(raw['csi_abs'],dtype=np.float32)
        targets=ref.start_seconds+np.arange(900)/30
        audit=audit_raw(t,values,raw.get('csi_complex'),targets,sample.x)
        audit.update(sample_id=sample.sample_id,recording_id=sample.recording_id,metadata=ref.metadata,
                     assessable=sample.assessable,quality_flags=list(sample.quality_flags),
                     raw_sha256=digest(ref.path),source_path=str(ref.path),
                     mat_fields=[field[0] for field in whosmat(ref.path)])
        audits.append(audit)
        if not audit['resampling_matches_loader']:
            raise ValueError('Independent interpolation does not match loader')
        if sample.assessable:
            if sample.sample_id not in by_id:
                raise ValueError('New quality-passing window outside existing evaluation')
            original=by_id[sample.sample_id]
            baseline=estimate_unified_rate(sample.x)['bpm']
            if baseline!=original['predictions']['unified']:
                raise ValueError('Frozen baseline mismatch')
            row={key:original[key] for key in ('sample_id','recording_id','participant_id','posture','room','environment','reference_bpm')}
            row.update(baseline_error=abs(baseline-original['reference_bpm']),predictions={'baseline':baseline},edits={})
            for c in configs:
                try:
                    x,edits=candidate_window(t,values,targets,sample.x,c)
                    estimate=estimate_unified_rate(x)
                    row['predictions'][c['name']]=estimate['bpm']
                    row['edits'][c['name']]={**edits,'available':True}
                except ValueError as error:
                    row['predictions'][c['name']]=None
                    row['edits'][c['name']]={'available':False,'reason':str(error)}
            rows.append(row)
        if (i+1)%25==0:
            print('Audited',i+1,'files; evaluated',len(rows),'quality-passing windows',flush=True)
    if set(by_id)!={r['sample_id'] for r in rows}:
        raise ValueError('Quality-passing evaluation set changed')
    (output/'raw_audit.json').write_text(json.dumps(audits,indent=2))
    (output/'window_predictions.json').write_text(json.dumps(rows,indent=2))
    folds,pooled={},[]
    for name,fold in protocol['folds'].items():
        records={part:{cache.manifest['samples'][i]['recording_id'] for i in fold[part]} for part in ('train','validation','assessment')}
        if records['validation']&records['assessment'] or records['train']&(records['validation']|records['assessment']):
            raise ValueError('Recording leakage')
        validation=[r for r in rows if r['recording_id'] in records['validation']]
        assessment=[r for r in rows if r['recording_id'] in records['assessment']]
        if any(r['participant_id']==fold['held_out_subject'] for r in validation) or any(r['participant_id']!=fold['held_out_subject'] for r in assessment):
            raise ValueError('Person leakage')
        chosen,inner=select_preprocessing(validation,methods)
        outcomes=[{**r,'predictions':{**r['predictions'],'selected':r['predictions'][chosen]},'selected_method':chosen} for r in assessment]
        pooled.extend(outcomes)
        folds[name]={'chosen':chosen,'inner_candidates':inner,'validation_ids':[r['sample_id'] for r in validation],
                      'assessment_ids':[r['sample_id'] for r in assessment],
                      'baseline':evaluate(outcomes,'baseline'),'selected':evaluate(outcomes,'selected')}
    if len(pooled)!=len(rows) or len({r['sample_id'] for r in pooled})!=len(rows):
        raise ValueError('Outer assessment must cover each source row exactly once')
    baseline,selected=evaluate(pooled,'baseline'),evaluate(pooled,'selected')
    signatures=Counter(a['peak_frame_hash'] for a in audits)
    audit_summary={'files':len(audits),'quality_passing':len(rows),
        'all_complex_available':all(a['complex_available'] for a in audits),
        'all_abs_complex_close':all(a['complex_abs_close'] for a in audits),
        'max_abs_complex_error':max(a['complex_abs_max_error'] or 0 for a in audits),
        'all_resampling_matches_loader':all(a['resampling_matches_loader'] for a in audits),
        'nonzero_start_offsets':sum(abs(a['start_offset_from_first_csi'])>1e-7 for a in audits),
        'total_adjacent_exact_duplicates':sum(a['adjacent_exact_duplicate_frames'] for a in audits),
        'files_with_raw_excursions':sum(a['raw_excursion_frames']>0 for a in audits),
        'files_with_resampled_excursions':sum(a['resampled_excursion_frames']>0 for a in audits),
        'shared_extreme_signatures':[{'hash':sig,'files':count,'recordings':len({a['recording_id'] for a in audits if a['peak_frame_hash']==sig})}
            for sig,count in signatures.most_common(10) if count>1],
        'mat_fields':sorted({key for a in audits for key in a['mat_fields']})}
    fixed={method:{'metrics':evaluate(rows,method),
         'person_macro_mae':float(np.mean([evaluate([r for r in rows if r['participant_id']==s],method)['mae_bpm'] for s in sorted({r['participant_id'] for r in rows})])),
         'by_person':{s:evaluate([r for r in rows if r['participant_id']==s],method) for s in sorted({r['participant_id'] for r in rows})},
         'new_large_errors':sum(r['baseline_error']<=5 and r['predictions'][method] is not None and abs(r['predictions'][method]-r['reference_bpm'])>5 for r in rows),
         'changed_rates':sum(r['predictions'][method]!=r['predictions']['baseline'] for r in rows),
         'edited_cells':sum(r['edits'].get(method,{}).get('replaced_cells',0) for r in rows)} for method in methods}
    safety=selected['retained']==len(rows) and selected['good_harm_rate']==0
    reduction=(baseline['large_errors']-selected['large_errors'])/max(1,baseline['large_errors'])
    per_person_nonworse=all(f['selected']['mae_bpm'] is not None and f['selected']['mae_bpm']<=f['baseline']['mae_bpm'] for f in folds.values())
    report={'config':config,'audit_summary':audit_summary,'baseline':baseline,'selected':selected,'folds':folds,
            'fixed_candidates_descriptive':fixed,'outer_safety_pass':safety,'large_error_reduction_fraction':reduction,
            'per_person_mae_nonworse':per_person_nonworse,'development_support_pass':safety and reduction>=.25 and per_person_nonworse,
            'application_enabled':False,'total_window_coverage':selected['retained']/len(dataset)}
    if any(digest(path)!=sha for path,sha in config['protected_artifact_hashes'].items()):
        raise ValueError('Protected application artifact changed')
    if any(digest(a['source_path'])!=a['raw_sha256'] for a in audits):
        raise ValueError('Raw source file changed')
    (output/'metrics.json').write_text(json.dumps(report,indent=2))
    (output/'selected_predictions.json').write_text(json.dumps(pooled,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('config','folds','fixed_candidates_descriptive')},indent=2))
    print('Selections:',{k:f['chosen'] for k,f in folds.items()})
    print('Fixed controls:',{k:{j:v['metrics'][j] for j in ('mae_bpm','large_errors','good_spoiled','good_rejected')} for k,v in fixed.items()})


if __name__=='__main__':
    main()
