"""Visualize frozen-model exploratory diagnostics; never select another threshold."""
import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

if __package__:
    from .preprocess import Preprocessor
    from .inference_runtime import sha256
else:
    from preprocess import Preprocessor
    from inference_runtime import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic-dir',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    args=parser.parse_args();p=args.diagnostic_dir
    a=json.loads((p/'diagnostics.json').read_text());rows=list(csv.DictReader((p/'sample_diagnostics.csv').open()))
    before=sha256(args.checkpoint)
    if before!=a['checkpoint_sha256_after']:raise ValueError('Not the audited frozen model')
    manifest={r['sample_id']:r for r in csv.DictReader(args.manifest.open())}
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True);pp=Preprocessor(state['preprocess_config'])
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    splits=['train_id','val_id','test_id'];pos=np.arange(3)
    for j,device in enumerate(['HP','ESP32']):
        ms=[a['metrics_by_split_device'][s][device] for s in splits]
        fpr=[m['fp']/(m['fp']+m['tn']) for m in ms]
        bars=axes[0].bar(pos+(j-.5)*.32,fpr,width=.32,label=device)
        for bar,m,rate in zip(bars,ms,fpr):
            axes[0].text(bar.get_x()+bar.get_width()/2,rate+.018,f'{m["fp"]}/{m["nonfall"]}\n{rate:.1%}',ha='center',fontsize=8)
    axes[0].set_xticks(pos,['Train','Validation','Test']);axes[0].set_ylim(0,1.07)
    axes[0].set_ylabel('False-positive rate: FP / actual Nonfall');axes[0].legend()
    axes[0].set_title('ESP32 errors already present on training data')
    rng=np.random.default_rng(0);colors={0:'#2878b5',1:'#e07b27'};labels=[]
    for i,(split,label) in enumerate((s,l) for s in splits for l in [0,1]):
        scores=[float(r['fall_score']) for r in rows if r['device']=='ESP32' and r['split']==split and r['label']==str(label)]
        axes[1].scatter(i+rng.uniform(-.17,.17,len(scores)),scores,s=10,alpha=.45,color=colors[label])
        axes[1].hlines(np.median(scores),i-.23,i+.23,color='black',lw=2)
        labels.append(f'{split.split("_")[0]}\n{"Fall" if label else "Nonfall"}\nn={len(scores)}')
    axes[1].axhline(a['threshold'],ls='--',color='red',label=f'Frozen threshold {a["threshold"]:.6f}')
    axes[1].set_xticks(range(6),labels,fontsize=8);axes[1].set_ylim(-.03,1.03)
    axes[1].set_ylabel('ESP32 fall score');axes[1].set_title('All ESP32 scores; black line = median');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.savefig(p/'device_error_scores.png',dpi=160);plt.close(fig)
    selected=[next(r for r in rows if r['device']=='ESP32' and r['split']=='test_id' and r['outcome']==o) for o in ['FP','TN','TP','FN']]
    fig,axes=plt.subplots(2,4,figsize=(15,7),layout='constrained');case_info=[]
    for i,row in enumerate(selected):
        m=manifest[row['sample_id']];x,mask=pp.transform_file(m['file_path'],'ESP32')
        with h5py.File(m['file_path']) as f:raw=f['CSI_amps'][()]
        for feature in [6,31,58]:axes[0,i].plot(x[feature],label=f'feature {feature}',lw=.8)
        axes[0,i].set_title(f'{row["outcome"]}: actual {"Fall" if row["label"]=="1" else "Nonfall"}\nscore={float(row["fall_score"]):.6f}')
        axes[0,i].set_ylabel('Normalized amplitude');axes[0,i].grid(alpha=.2);axes[0,i].legend(fontsize=7)
        axes[1,i].imshow(x,aspect='auto',cmap='coolwarm',vmin=-3,vmax=3)
        axes[1,i].axhline(63.5,color='black',ls='--',lw=1)
        axes[1,i].text(245,145,'168 padded rows = 0',ha='center',fontsize=9)
        axes[1,i].set_xlabel('Time sample index');axes[1,i].set_ylabel('Input feature position')
        case_info.append({'sample_id':row['sample_id'],'outcome':row['outcome'],'fall_score':float(row['fall_score']),
                          'file':m['file_path'],'raw_shape':list(raw.shape),'features_plotted':[6,31,58],
                          'selection':'first metadata-order ESP32 test record in each confusion category'})
    fig.suptitle('Frozen-model cases: curves and actual padded input; no parameter changes')
    fig.savefig(p/'esp32_error_cases.png',dpi=160);plt.close(fig)
    (p/'case_selection.json').write_text(json.dumps(case_info,indent=2))
    result={}
    for split in splits:
        result[split]={}
        for device in ['HP','ESP32']:
            subset=[r for r in rows if r['split']==split and r['device']==device]
            positive=np.array([float(r['fall_score']) for r in subset if r['label']=='1'])
            negative=np.array([float(r['fall_score']) for r in subset if r['label']=='0'])
            diff=positive[:,None]-negative[None,:]
            result[split][device]={'roc_auc_posthoc':float(((diff>0).sum()+.5*(diff==0).sum())/diff.size),
                                   'positive_score_median':float(np.median(positive)),
                                   'negative_score_median':float(np.median(negative))}
    (p/'ranking_diagnostics.json').write_text(json.dumps(result,indent=2))
    assert sha256(args.checkpoint)==before
    print('Saved two figures, four fixed-rule cases and post-hoc ranking summaries.')


if __name__=='__main__':main()
