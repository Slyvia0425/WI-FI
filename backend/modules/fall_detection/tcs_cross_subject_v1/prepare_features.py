"""Prepare train/validation GCV and train-only statistics with bounded disk/RAM.

Test subject04 is not read by this stage. Labels for CT windows require their
entire actual timestamp span to end within the documented first120 seconds.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

HERE=Path(__file__).resolve().parent
MODULE=HERE.parent
sys.path.insert(0,str(MODULE/'tcs_withinsubject_v1'))
from prepare import transform_dat_bytes, RECIPE, sha
from replay_continuous import decode_amplitude, make_feature


def write_json(path,data):
    path.write_text(json.dumps(data,indent=2))


def audited_bytes(archive,row):
    data=archive.read(row['archive_member'])
    if hashlib.sha256(data).hexdigest()!=row['raw_file_sha256']:
        raise ValueError('Record changed after audit: '+row['archive_member'])
    if row.get('decoded_complete_prefix_bytes'):
        prefix=data[:row['decoded_complete_prefix_bytes']]
        if hashlib.sha256(prefix).hexdigest()!=row['decoded_complete_prefix_sha256']:
            raise ValueError('Reviewed prefix SHA mismatch')
        data=prefix
    return data


def fit_statistics(cache,rows,include_ct,cache_sha):
    indices=[i for i,r in enumerate(rows) if r['role']=='train' and (include_ct or r['kind']=='single_action')]
    sums=np.zeros(90,dtype=np.float64);squares=np.zeros(90,dtype=np.float64);n=0
    for first in range(0,len(indices),16):
        block=np.asarray(cache[indices[first:first+16]],dtype=np.float64)
        sums+=block.sum(axis=(0,1,2));squares+=np.square(block).sum(axis=(0,1,2));n+=block.shape[0]*400
    mean=sums/n;std=np.sqrt(np.maximum(squares/n-mean*mean,0))
    constant=std<1e-8;scale=np.where(constant,1.,std)
    return {'fit_role':'train','fit_subjects':[1,2],'includes_continuous_negatives':include_ct,
            'fit_sample_count':len(indices),'time_points_per_feature':n,
            'axes':'per native Rx/subcarrier feature over selected training samples and400 GCV time positions',
            'weighting':'each selected training window/timepoint equal; overlapping windows are correlated',
            'ddof':0,'small_std_threshold':1e-8,'small_std_policy':'center and divide by1; no division by zero',
            'constant_feature_indices':np.flatnonzero(constant).tolist(),'mean':mean.tolist(),'std':std.tolist(),'scale':scale.tolist(),
            'feature_cache_sha256':cache_sha,
            'scope':'our training-only standardization variant; missing official preprocessing statistics scope is unresolved'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--eligibility',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError('Use a fresh feature output directory')
    eligibility=json.loads((args.eligibility/'eligibility.json').read_text())
    if sha(args.eligibility/'records.json')!=eligibility['records_sha256']:
        raise ValueError('Eligibility records changed')
    audited=json.loads((args.eligibility/'records.json').read_text())
    selected=sorted([r for r in audited if r['input_eligibility'].startswith('eligible') and r['role'] in ('train','validation')],key=lambda r:r['archive_member'])
    if any(r['nominal_subject_id']==4 for r in selected):raise ValueError('Test inclusion')
    max_rows=sum(1 if r['source_kind']=='single_action' else 113 for r in selected)
    required=max_rows*400*90*4+100*1024**2
    free=shutil.disk_usage(args.output.parent if args.output.parent.exists() else MODULE).free
    if free<required+512*1024**2:raise RuntimeError('Insufficient disk for feature cache plus512MiB reserve')
    args.output.mkdir(parents=True)
    b,a=signal.butter(8,0.2,'lowpass')
    samples=[];record_details=[];demos=[];minimum=float('inf');maximum=float('-inf')
    first_check=None;clock=time.perf_counter()
    output_file=args.output/'features.f32'
    with output_file.open('wb') as out:
        for info in eligibility['archives']:
            if info['role']=='test':continue
            path=Path(info['path'])
            if sha(path)!=info['sha256']:raise ValueError('Archive changed')
            group=[r for r in selected if r['nominal_subject_id']==info['subject']]
            with zipfile.ZipFile(path) as archive:
                for index,row in enumerate(group,1):
                    data=audited_bytes(archive,row)
                    base={'source_member':row['archive_member'],'subject':row['nominal_subject_id'],
                          'activity':row['activity_id'],'role':row['role'],'source_sha256':row['raw_file_sha256']}
                    if row['source_kind']=='single_action':
                        feature,details=transform_dat_bytes(data)
                        local=[(0,feature,0.,details['selected_timestamp_span_seconds'])]
                        if not demos and row['single_action_binary_label']==1:
                            amp,elapsed=decode_amplitude(data[:8000*575]);demos.append(('Fall SA '+row['archive_member'],elapsed,amp))
                    else:
                        if row['annotation_record']['recorded_fall_time_seconds']<=120:
                            raise ValueError('Early fall annotation conflicts with negative interval')
                        amp,elapsed=decode_amplitude(data)
                        starts=np.arange(0,len(amp)-8000+1,1000,dtype=int)
                        starts=starts[elapsed[starts+7999]<=120.0]
                        if not len(starts):raise ValueError('No complete background windows')
                        local=[]
                        for start in starts:
                            feature=make_feature(amp[start:start+8000],b,a)
                            local.append((int(start),feature,float(elapsed[start]),float(elapsed[start+7999])))
                        if first_check is None:
                            reference,_=transform_dat_bytes(data[:8000*575])
                            difference=float(np.max(np.abs(reference-local[0][1])))
                            if difference>1e-7:raise ValueError('Continuous transform differs from frozen SA transform')
                            first_check={'member':row['archive_member'],'max_abs_feature_difference':difference,'tolerance':1e-7}
                        details={'source_packets':len(amp),'background_windows':len(local),
                                 'latest_window_end_seconds':local[-1][3],'fall_annotation_seconds':row['annotation_record']['recorded_fall_time_seconds']}
                        if len(demos)==1 and row['role']=='train' and row['activity_id']==12:
                            demos.append(('Nonfall walking '+row['archive_member'],elapsed[:8000].copy(),amp[:8000].copy()))
                        del amp,elapsed
                    for start,feature,t0,t1 in local:
                        if feature.shape!=(1,400,90) or not np.isfinite(feature).all():raise ValueError('Invalid feature values/shape')
                        out.write(np.ascontiguousarray(feature,dtype=np.float32).tobytes())
                        kind='single_action' if row['source_kind']=='single_action' else 'continuous_background'
                        label=row['single_action_binary_label'] if kind=='single_action' else 0
                        samples.append({**base,'cache_index':len(samples),'sample_id':row['archive_member']+f':packet{start}',
                                        'kind':kind,'label':label,'start_packet':start,'end_packet':start+7999,
                                        'start_seconds':t0,'end_seconds':t1})
                        minimum=min(minimum,float(feature.min()));maximum=max(maximum,float(feature.max()))
                    record_details.append({**base,**details})
                    del data,local
                    if index%30==0 or row['source_kind']!='single_action':
                        print(json.dumps({'subject':info['subject'],'records_prepared':index,'total_records':len(group),
                                          'feature_rows':len(samples),'elapsed_seconds':round(time.perf_counter()-clock,1)}),flush=True)
    shape=[len(samples),1,400,90]
    if output_file.stat().st_size!=int(np.prod(shape))*4:raise ValueError('Feature cache byte count mismatch')
    cache_sha=sha(output_file)
    write_json(args.output/'samples.json',samples)
    cache=np.memmap(output_file,dtype=np.float32,mode='r',shape=tuple(shape))
    norm_checks=[]
    for include_ct,name in ((False,'normalization_sa.json'),(True,'normalization_sa_ct.json')):
        norm=fit_statistics(cache,samples,include_ct,cache_sha);write_json(args.output/name,norm)
        mu=np.asarray(norm['mean']);scale=np.asarray(norm['scale']);total=np.zeros(90);squared=np.zeros(90);n=0
        for i,r in enumerate(samples):
            if r['role']=='train' and (include_ct or r['kind']=='single_action'):
                transformed=(cache[i].astype(np.float64)-mu)/scale
                if not np.isfinite(transformed).all():raise ValueError('Nonfinite normalized training features')
                total+=transformed.sum(axis=(0,1));squared+=np.square(transformed).sum(axis=(0,1));n+=400
        residual=total/n;variance=squared/n-residual*residual;variable=np.asarray(norm['std'])>=1e-8
        norm_checks.append({'file':name,'train_sample_count':norm['fit_sample_count'],
                            'max_abs_training_mean':float(np.abs(residual).max()),
                            'max_abs_training_std_minus1_variable_features':float(np.abs(np.sqrt(variance[variable])-1).max()),
                            'constant_features':norm['constant_feature_indices']})
    counts=Counter((r['role'],r['kind'],r['label']) for r in samples)
    summary={'shape':shape,'feature_cache_sha256':cache_sha,'sample_manifest_sha256':sha(args.output/'samples.json'),
             'cache_bytes':output_file.stat().st_size,'feature_range':[minimum,maximum],'finite_ratio':1.0,
             'counts':[{'role':k[0],'kind':k[1],'label':k[2],'count':v} for k,v in sorted(counts.items())],
             'eligibility_sha256':sha(args.eligibility/'eligibility.json'),'recipe':RECIPE,
             'background_policy':'whole8000-packet window ends at actual elapsed<=120s; no resampling/padding',
             'continuous_transform_reference_check':first_check,'normalization_checks':norm_checks,
             'test_subject_features_prepared':False,'test_model_evaluated':False,'training_run':False,
             'elapsed_seconds':time.perf_counter()-clock,'source_code_sha256':sha(Path(__file__)),
             'records':record_details}
    write_json(args.output/'preparation.json',summary)
    # Actual sample/batch check uses the same memory-mapped Dataset planned for training.
    sys.path.insert(0,str(HERE))
    from dataset import GCVDataset
    from torch.utils.data import DataLoader
    ds=GCVDataset(args.output,'train',True,args.output/'normalization_sa_ct.json')
    bx,by,ids=next(iter(DataLoader(ds,batch_size=8,shuffle=False,num_workers=0)))
    check={'sample_shape':list(ds[0][0].shape),'batch_shape':list(bx.shape),'label_dtype':str(by.dtype),
           'batch_labels':by.tolist(),'batch_range':[float(bx.min()),float(bx.max())],
           'all_batch_finite':bool(np.isfinite(bx.numpy()).all()),'ids':list(ids),
           'label_convention':'Fall1, Nonfall0; int64 for the two-logit CrossEntropy CNN'}
    write_json(args.output/'dataset_check.json',check)
    fig,axes=plt.subplots(2,1,figsize=(11,6),constrained_layout=True)
    for ax,(title,elapsed,amp) in zip(axes,demos):
        for feature_index in (3,15,33,45,63):ax.plot(elapsed,amp[:,feature_index],lw=.6,label=f'feature{feature_index}')
        ax.set_title(title);ax.set_xlabel('Elapsed CSI time (s)');ax.set_ylabel('RF-scaled amplitude');ax.legend(ncol=5,fontsize=8)
    fig.savefig(args.output/'input_examples.png',dpi=150);plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('records','recipe')},indent=2),flush=True)
    print(json.dumps(check,indent=2),flush=True)


if __name__=='__main__':main()
