"""Post-test exploratory error audit. Does not modify model, normalization or threshold."""
import argparse
import collections
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import DataLoader

if __package__:
    from .audit_data import digest
    from .dataset import FallDataset
    from .inference_runtime import collect_predictions,load_model,record_metrics,sha256
    from .preprocess import Preprocessor
else:
    from audit_data import digest
    from dataset import FallDataset
    from inference_runtime import collect_predictions,load_model,record_metrics,sha256
    from preprocess import Preprocessor


def distribution(values):
    x=np.array(values,dtype=float)
    return dict(zip(['min','q25','median','q75','max'],np.quantile(x,[0,.25,.5,.75,1]).tolist())) if len(x) else None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--stage5-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    if (args.output_dir/'diagnostics.json').exists():
        raise FileExistsError('Diagnostic run exists; choose a new output directory')
    before=sha256(args.checkpoint)
    model,state=load_model(args.checkpoint,'cpu');pp=Preprocessor(state['preprocess_config']);cfg=pp.config
    threshold=state['selected_threshold']
    rows=list(csv.DictReader(args.manifest.open()));byid={r['sample_id']:r for r in rows}
    predictions={}
    for split,file in [('val_id',args.stage5_dir/'reload_validation/val_id_predictions.csv'),
                       ('test_id',args.stage5_dir/'official_test/test_id_predictions.csv')]:
        records=list(csv.DictReader(file.open()))
        for r in records:r['fall_score']=float(r['fall_score']);r['label']=int(r['label'])
        predictions[split]=records
    train=FallDataset(args.manifest,cfg,'train_id')
    predictions['train_id']=collect_predictions(model,DataLoader(train,batch_size=32,shuffle=False),'cpu')
    print('Frozen-model training predictions collected; weights unchanged.',flush=True)
    pred_byid={r['sample_id']:r for records in predictions.values() for r in records}
    for r in rows:
        if r['sample_id'] not in pred_byid:raise ValueError('Missing primary-split prediction')
    # All source fields below are read from the audited manifest.
    raw_stats=[];independent={};issues=[]
    max_adaptation_error=0.;nonzero_padding=0;unexpected_constant_points=0;hash_mismatches=0
    for i,r in enumerate(rows,1):
        with h5py.File(r['file_path']) as f:
            keys=list(f.keys());raw=f['CSI_amps'][()]
        if keys!=['CSI_amps'] or not np.isfinite(raw).all() or raw.dtype!=np.float32:
            issues.append({'sample_id':r['sample_id'],'reason':'key/dtype/finite mismatch'})
        if digest(raw)!=r['content_hash']:hash_mismatches+=1
        a=raw[:,:,0].astype(np.float64);device=r['device'];params=cfg['devices'][device]
        if a.shape!=(params['features'],500):raise ValueError('Source axes/shape mismatch')
        mean=np.array(params['mean']);scale=np.array(params['scale']);std=np.array(params['std'])
        active=std>=cfg['epsilon'];constant=~active
        x,mask=pp.transform_file(r['file_path'],device)
        expected=(a-mean[:,None])/scale[:,None]
        error=float(np.max(np.abs(x[:len(a)].astype(float)-expected)))
        max_adaptation_error=max(max_adaptation_error,error)
        if not np.array_equal(mask,np.arange(232)<len(a)):raise ValueError('Mask mismatch')
        if not np.isfinite(x).all():raise ValueError('Nonfinite transformed sample')
        nonzero_padding+=int(np.count_nonzero(x[len(a):]))
        if constant.any():unexpected_constant_points+=int(np.count_nonzero(a[constant]-mean[constant,None]))
        if 'train_id' in r['split'].split(';'):
            st=independent.setdefault(device,{'count':0,'sum':np.zeros(len(a)),'sum_sq':np.zeros(len(a))})
            st['count']+=a.shape[1];st['sum']+=a.sum(axis=1);st['sum_sq']+=(a*a).sum(axis=1)
        v=a[active];z=expected[active]
        centered=v-v.mean(axis=1,keepdims=True)
        denom=np.sqrt((centered[:,:-1]**2).sum(axis=1)*(centered[:,1:]**2).sum(axis=1))
        lag=np.divide((centered[:,:-1]*centered[:,1:]).sum(axis=1),denom,
                      out=np.zeros(len(v)),where=denom>1e-12)
        pred=pred_byid[r['sample_id']];is_fall=pred['fall_score']>=threshold;label=int(r['binary_label'])
        outcome=('TP' if is_fall else 'FN') if label else ('FP' if is_fall else 'TN')
        split=next(s for s in r['split'].split(';') if s in ['train_id','val_id','test_id'])
        raw_stats.append({'sample_id':r['sample_id'],'file_path':r['file_path'],'split':split,'device':device,'user':r['user'],
                          'environment':r['environment'],'session':r['session'],'label':label,
                          'fall_score':pred['fall_score'],'outcome':outcome,
                          'raw_active_mean':float(v.mean()),'raw_active_std':float(v.std()),
                          'raw_zero_fraction':float((a==0).mean()),
                          'raw_active_temporal_std_median':float(np.median(v.std(axis=1))),
                          'raw_active_lag1_median':float(np.median(lag)),
                          'z_active_mean':float(z.mean()),'z_active_rms':float(np.sqrt((z*z).mean())),
                          'z_abs_gt5_fraction':float((np.abs(z)>5).mean()),
                          'z_temporal_diff_rms':float(np.sqrt((np.diff(z,axis=1)**2).mean())),
                          'zero_padding_features':232-len(a),'adaptation_max_abs_error':error})
        if i%1000==0:print('Raw and transformed audit',i,'/',len(rows),flush=True)
    norm_check={}
    for device,st in independent.items():
        mean=st['sum']/st['count'];std=np.sqrt(np.maximum(0,st['sum_sq']/st['count']-mean*mean))
        p=cfg['devices'][device]
        norm_check[device]={'independent_method':'float64 raw sums and sums-of-squares; original fit used merged centered M2',
             'observations_per_feature':st['count'],
             'max_abs_mean_error':float(np.max(np.abs(mean-np.array(p['mean'])))),
             'max_abs_std_error':float(np.max(np.abs(std-np.array(p['std'])))),
             'minimum_nonconstant_std':float(min(s for s in p['std'] if s>=cfg['epsilon'])),
             'constant_features':p['constant_training_features']}
        assert norm_check[device]['max_abs_mean_error']<1e-8 and norm_check[device]['max_abs_std_error']<1e-8
    assert nonzero_padding==0 and unexpected_constant_points==0 and hash_mismatches==0 and not issues
    result={'scope':'post-test exploratory diagnosis; no threshold selection/retraining',
            'threshold':threshold,'checkpoint_sha256_before':before,'normalization':norm_check,
            'raw_records_audited':len(rows),'content_hash_mismatches':hash_mismatches,
            'nonzero_padding_points':nonzero_padding,'unexpected_constant_feature_points':unexpected_constant_points,
            'max_float32_adaptation_abs_error':max_adaptation_error,'issues':issues,
            'metrics_by_split_device':{},'esp32_test_by_user':{},'signal_summaries':{}}
    for split,records in predictions.items():
        result['metrics_by_split_device'][split]={d:record_metrics([r for r in records if r['device']==d],threshold)
                                                  for d in ['HP','ESP32']}
        with (args.output_dir/(split+'_predictions.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    fields=['fall_score','raw_active_mean','raw_active_temporal_std_median','z_active_mean',
            'z_active_rms','z_abs_gt5_fraction','z_temporal_diff_rms','raw_active_lag1_median']
    for split in ['train_id','val_id','test_id']:
        for label in [0,1]:
            subset=[r for r in raw_stats if r['device']=='ESP32' and r['split']==split and r['label']==label]
            result['signal_summaries'][f'ESP32/{split}/{label}']={k:distribution([r[k] for r in subset]) for k in fields}
            for outcome in ['FP','TN'] if label==0 else ['TP','FN']:
                group=[r for r in subset if r['outcome']==outcome]
                result['signal_summaries'][f'ESP32/{split}/{outcome}']={'n':len(group),**{k:distribution([r[k] for r in group]) for k in fields}}
    esp_test=[r for r in predictions['test_id'] if r['device']=='ESP32']
    for user in sorted({byid[r['sample_id']]['user'] for r in esp_test}):
        result['esp32_test_by_user'][user]=record_metrics([r for r in esp_test if byid[r['sample_id']]['user']==user],threshold)
    result['training_composition']={'ESP32_fraction':len([r for r in train.rows if r['device']=='ESP32'])/len(train),
                                   'devices':{d:dict(collections.Counter(r['original_label'] for r in train.rows if r['device']==d)) for d in ['HP','ESP32']}}
    result['esp32_environments']=sorted({r['environment'] for r in rows if r['device']=='ESP32'})
    result['esp32_users_by_split']={s:sorted({r['user'] for r in rows if r['device']=='ESP32' and s in r['split'].split(';')}) for s in predictions}
    with (args.output_dir/'sample_diagnostics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(raw_stats[0]));w.writeheader();w.writerows(raw_stats)
    esp_errors=[r for r in raw_stats if r['device']=='ESP32' and r['split']=='test_id' and r['outcome'] in ['FP','FN']]
    with (args.output_dir/'esp32_test_errors.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(raw_stats[0]));w.writeheader();w.writerows(esp_errors)
    result['checkpoint_sha256_after']=sha256(args.checkpoint)
    assert result['checkpoint_sha256_after']==before
    (args.output_dir/'diagnostics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ['normalization','metrics_by_split_device','esp32_test_by_user',
                                           'max_float32_adaptation_abs_error','nonzero_padding_points']},indent=2),flush=True)


if __name__=='__main__':main()
