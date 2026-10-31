"""Read-only comparison of best checkpoints, official val_id only, threshold 0.5."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import FallDataset
from models.cnn import FallCNN
from models.sensefi_style import SenseFiStyleCNN, collate_device_segments
from preprocess import manifest_rows, read_amplitude

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'reports/validation_model_comparison_v2'
PATHS = {'original': ROOT/'outputs/cnn_seed42_mps_v1/best.pt',
         'new': ROOT/'outputs/sensefi_style_seed42_mps_v1/best.pt'}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def metrics(rows, name):
    counts = Counter(r[name+'_outcome'] for r in rows)
    tp,fp,fn,tn = (counts[k] for k in ('TP','FP','FN','TN'))
    return dict(n=len(rows), fall=tp+fn, nonfall=fp+tn, tp=tp, fp=fp, fn=fn, tn=tn,
                precision=tp/(tp+fp) if tp+fp else 0, recall=tp/(tp+fn) if tp+fn else 0,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
                false_positive_rate=fp/(fp+tn) if fp+tn else 0)

def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def main():
    if OUT.exists():
        raise FileExistsError('Preserve earlier comparison; choose a new output path')
    OUT.mkdir(parents=True)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    before={str(p):sha(p) for p in [*PATHS.values(), ROOT/'outputs/cnn_seed42_mps_v1/frozen.pt',
        ROOT/'outputs/cnn_seed42_mps_v1/frozen.json', ROOT/'outputs/preprocess_v1.json',
        ROOT/'models/cnn.py',ROOT/'models/sensefi_style.py',ROOT/'preprocess.py',ROOT/'dataset.py']}
    checkpoints={n:torch.load(p,map_location='cpu',weights_only=True) for n,p in PATHS.items()}
    a,b=checkpoints.values()
    assert a['preprocess_config']==b['preprocess_config']
    assert a['split_identifiers']==b['split_identifiers']
    manifest=ROOT/'reports/sample_manifest.csv'
    assert sha(manifest)==a['manifest_sha256']==b['manifest_sha256']
    all_rows=manifest_rows(manifest)
    metadata={r['sample_id']:r for r in all_rows if 'val_id' in r['split'].split(';')}
    results={}; replays={}
    for name,c in checkpoints.items():
        dataset=FallDataset(manifest,c['preprocess_config'],'val_id')
        model=FallCNN(**c['model_config']) if name=='original' else SenseFiStyleCNN(dropout=c['model_config']['dropout'])
        model.load_state_dict(c['model_state_dict'],strict=True);model.eval()
        loader=DataLoader(dataset,batch_size=32,shuffle=False,
                          collate_fn=collate_device_segments if name=='new' else None)
        rows={}
        with torch.no_grad():
            for batch in loader:
                scores=model(batch['x']).squeeze(-1).sigmoid().tolist()
                for sid,d,y,s in zip(batch['sample_id'],batch['device'],batch['y'].tolist(),scores):
                    assert np.isfinite(s) and sid not in rows and int(y)==int(metadata[sid]['binary_label'])
                    rows[sid]=dict(score=s,device=d,label=int(y))
        assert set(rows)==set(metadata)==set(c['split_identifiers']['val_id'])
        saved_dir='stage4_seed42_mps_v1' if name=='original' else 'sensefi_style_seed42_mps_v1'
        saved={r['sample_id']:r for r in csv.DictReader((ROOT/'reports'/saved_dir/'best_validation_predictions.csv').open())}
        maxdiff=max(abs(rows[s]['score']-float(saved[s]['fall_score'])) for s in rows)
        changed=sum((rows[s]['score']>=.5)!=(float(saved[s]['fall_score'])>=.5) for s in rows)
        assert maxdiff<1e-5 and changed==0
        replays[name]=dict(best_epoch=c['best_epoch'],max_score_difference_from_mps=maxdiff,classifications_changed=changed)
        results[name]=rows
    joined=[]
    for sid,r in metadata.items():
        row={k:r[k] for k in ('sample_id','device','user','environment','session','file_path','content_hash','shape','dtype')}
        y=int(r['binary_label']);row['label']=y
        for name in results:
            v=results[name][sid];assert v['device']==r['device']
            pred=int(v['score']>=.5)
            row[name+'_score']=v['score'];row[name+'_outcome']=('TP' if pred else 'FN') if y else ('FP' if pred else 'TN')
        oldbad=row['original_outcome'] in ('FP','FN');newbad=row['new_outcome'] in ('FP','FN')
        row['error_group']='both_wrong' if oldbad and newbad else 'original_only' if oldbad else 'new_only' if newbad else 'both_correct'
        joined.append(row)
    esp=[r for r in joined if r['device']=='ESP32']
    grouped={g:[r for r in esp if r['error_group']==g] for g in ('both_wrong','original_only','new_only','both_correct')}
    summary={'scope':'official val_id only; descriptive exploratory comparison; no model selection',
             'threshold':.5,'compute_device':'cpu','same_normalization':True,'same_splits':True,
             'replay_checks':replays,'test_evaluated':False,'training_performed':False,
             'metrics':{d:{n:metrics([r for r in joined if d=='all' or r['device']==d],n) for n in results} for d in ('all','HP','ESP32')},
             'esp32_groups':{g:{'n':len(rows),'fall':sum(r['label'] for r in rows),
                'nonfall':sum(1-r['label'] for r in rows),'users':dict(Counter(r['user'] for r in rows))} for g,rows in grouped.items()}}
    # Reopen every ESP32 validation segment. No test HDF5 is opened.
    constant=a['preprocess_config']['devices']['ESP32']['constant_training_features']
    for r in esp:
        raw=read_amplitude(r['file_path']);assert raw.shape==(64,500)
        assert np.isfinite(raw).all() and (raw[constant]==0).all()
        active=np.delete(raw,constant,axis=0)
        r['raw_active_mean']=float(active.mean())
        r['raw_active_temporal_std_median']=float(np.median(active.std(axis=1)))
    summary['esp32_data_checks']={'reopened_n':len(esp),'shape':[64,500], 'finite':True,'training_constant_positions_all_zero':True,
          'environments':dict(Counter(r['environment'] for r in esp))}
    summary['group_signal_descriptions']={g:{k:float(np.median([r[k] for r in rows])) for k in ('raw_active_mean','raw_active_temporal_std_median')} for g,rows in grouped.items() if rows}
    # Content overlap is evidence, not automatic proof that a shared parent recording exists.
    train_hashes={r['content_hash'] for r in all_rows if 'train_id' in r['split'].split(';')}
    summary['esp32_validation_content_matches_train']=[r['sample_id'] for r in esp if r['content_hash'] in train_hashes]
    write_csv(OUT/'validation_predictions_comparison.csv',joined)
    write_csv(OUT/'esp32_all_records.csv',esp)
    errors=[r for r in esp if r['error_group']!='both_correct']
    write_csv(OUT/'esp32_errors.csv',errors)
    examples=[]
    for group,rows in grouped.items():
        for label in (0,1):
            candidates=sorted((r for r in rows if r['label']==label),key=lambda r:r['sample_id'])
            if candidates:examples.append(candidates[0])
    summary['examples_selection']='First sample ID alphabetically per error group and true class, including correct controls; not selected by appearance'
    summary['examples']=examples
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for y,label,color in [(0,'Nonfall','tab:blue'),(1,'Fall','tab:orange')]:
        rr=[r for r in esp if r['label']==y]
        axes[0].scatter([r['original_score'] for r in rr],[r['new_score'] for r in rr],label=label,color=color,alpha=.75)
    axes[0].axvline(.5,color='gray',ls='--');axes[0].axhline(.5,color='gray',ls='--')
    axes[0].set(xlabel='Original fall score',ylabel='New fall score',xlim=(-.02,1.02),ylim=(-.02,1.02),title='Same 79 ESP32 validation segments');axes[0].legend()
    labels=list(grouped)
    axes[1].bar(labels,[sum(r['label']==0 for r in grouped[g]) for g in labels],label='Nonfall')
    axes[1].bar(labels,[sum(r['label']==1 for r in grouped[g]) for g in labels],bottom=[sum(r['label']==0 for r in grouped[g]) for g in labels],label='Fall')
    axes[1].tick_params(axis='x',rotation=20);axes[1].set_ylabel('Segments');axes[1].legend()
    fig.savefig(OUT/'esp32_comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(len(examples),1,figsize=(11,2.3*len(examples)),layout='constrained',squeeze=False)
    for ax,r in zip(axes[:,0],examples):
        raw=read_amplitude(r['file_path'])
        for f in (6,31,58):ax.plot(raw[f],label=f'position {f}',lw=.8)
        ax.set_title(f"{r['error_group']} | {'Fall' if r['label'] else 'Nonfall'} | {r['sample_id']}\nOriginal {r['original_score']:.4f} ({r['original_outcome']}), new {r['new_score']:.4f} ({r['new_outcome']})",fontsize=9)
        ax.set_xlabel('Time index (not seconds)');ax.set_ylabel('Raw amplitude');ax.legend(loc='upper right',fontsize=7)
    fig.savefig(OUT/'esp32_example_traces.png',dpi=140);plt.close(fig)
    assert before=={s:sha(Path(s)) for s in before}
    summary['preserved_artifacts_sha256']=before
    (OUT/'comparison.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('examples','preserved_artifacts_sha256')},indent=2))
    print('EXAMPLES',json.dumps([{k:r[k] for k in ('sample_id','label','error_group','original_score','new_score','original_outcome','new_outcome')} for r in examples],indent=2))

if __name__=='__main__':main()
