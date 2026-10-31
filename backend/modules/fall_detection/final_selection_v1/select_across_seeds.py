"""Choose a candidate using only predeclared seeds and official validation predictions."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from compare_thresholds import load_validation,metrics,select_threshold,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol',type=Path,required=True)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--run',action='append',required=True,help='candidate:seed:validation.csv:checkpoint.pt')
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args();protocol=json.loads(args.protocol.read_text())
    if args.output_dir.exists():raise FileExistsError('Fresh output directory required')
    expected={r['sample_id']:r for r in csv.DictReader(args.manifest.open()) if 'val_id' in r['split'].split(';')}
    runs={};flat=[]
    for item in args.run:
        kind,seed,predfile,ckfile=item.split(':',3);seed=int(seed)
        if kind not in protocol['candidates'] or seed not in protocol['seeds'] or (kind,seed) in runs:raise ValueError('Unexpected/duplicate candidate seed')
        checkpoint_state=torch.load(ckfile,map_location='cpu',weights_only=True)
        assert checkpoint_state['seed']==seed, 'Checkpoint seed does not match declared run'
        assert set(checkpoint_state['split_identifiers']['val_id'])==set(expected)
        if kind=='original_1d_cnn':assert checkpoint_state['model_version']=='cnn_baseline_v1'
        else:assert checkpoint_state['model']=='ResNet18Classifier'
        records=load_validation(Path(predfile),expected);selected,_=select_threshold(records)
        run={'kind':kind,'seed':seed,'checkpoint':str(Path(ckfile).resolve()),'checkpoint_sha256':sha(Path(ckfile)),
             'prediction_file':str(Path(predfile).resolve()),'predictions_sha256':sha(Path(predfile)),
             'selected_threshold':selected['threshold'],'groups':{}}
        for group in ('all','HP','ESP32'):
            subset=records if group=='all' else [r for r in records if r['device']==group]
            run['groups'][group]={point:metrics(subset,t) for point,t in [('fixed_0_5',.5),('selected_global',selected['threshold'])]}
            for point,m in run['groups'][group].items():flat.append({'kind':kind,'seed':seed,'group':group,'operating_point':point,'threshold':.5 if point=='fixed_0_5' else selected['threshold'],**m})
        runs[kind,seed]=run
    assert set(runs)=={(kind,seed) for kind in protocol['candidates'] for seed in protocol['seeds']},'All six runs are required before selection'
    for kind in protocol['candidates']:
        assert len({runs[kind,s]['checkpoint_sha256'] for s in protocol['seeds']})==len(protocol['seeds']), 'Different seeds must not reuse one checkpoint'
    summary={}
    for kind in protocol['candidates']:
        summary[kind]={}
        for group in ('all','HP','ESP32'):
            summary[kind][group]={}
            for point in ('fixed_0_5','selected_global'):
                summary[kind][group][point]={metric:{'mean':float(np.mean(values)),'sample_std':float(np.std(values,ddof=1))} for metric in ('precision','recall','fall_f1','false_positive_rate') for values in [[runs[kind,s]['groups'][group][point][metric] for s in protocol['seeds']]]}
    def rank(kind):
        m=summary[kind]['all']['selected_global']
        return (m['fall_f1']['mean'],m['recall']['mean'],1 if kind=='original_1d_cnn' else 0)
    selected_kind=max(protocol['candidates'],key=rank)
    selected_run=runs[selected_kind,protocol['final_artifact_seed']]
    paired=[runs['original_1d_cnn',s]['groups']['all']['selected_global']['fall_f1']-runs['official_resnet18',s]['groups']['all']['selected_global']['fall_f1'] for s in protocol['seeds']]
    result={'selection_scope':'official val_id only','test_used_for_selection':False,'protocol_sha256':sha(args.protocol),
            'manifest_sha256':sha(args.manifest),'summary':summary,'runs':[runs[k,s] for k in protocol['candidates'] for s in protocol['seeds']],
            'selected_kind':selected_kind,'final_artifact_seed':protocol['final_artifact_seed'],'selected_run':selected_run,
            'paired_f1_difference_original_minus_resnet':paired,'known_limitations':protocol['known_limitations'],
            'prior_test_exposure':protocol['test_policy']['prior_test_exposure'],'deployment_claim':protocol['deployment_claim']}
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'selection.json').write_text(json.dumps(result,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    for ax,group in zip(axes,('all','HP','ESP32')):
        for j,kind in enumerate(protocol['candidates']):
            for offset,point,marker in [(-.1,'fixed_0_5','o'),(.1,'selected_global','s')]:
                stats=summary[kind][group][point]['fall_f1']
                ax.errorbar(j+offset,stats['mean'],yerr=stats['sample_std'],fmt=marker,capsize=4,
                            color=('tab:blue' if point=='fixed_0_5' else 'tab:orange'),
                            label=point if j==0 else None)
        ax.set(xticks=[0,1],xticklabels=['Original 1D-CNN','Official ResNet18'],
               title=group+' validation',ylabel='Fall F1 (mean ± sample SD)',ylim=(0,1.05))
        ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('3 seeds on shared val_id; SD describes training variability, not new-data uncertainty')
    fig.savefig(args.output_dir/'three_seed_comparison.png',dpi=160);plt.close(fig)
    with (args.output_dir/'all_seed_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    print(json.dumps({'selected_kind':selected_kind,'seed':42,'threshold':selected_run['selected_threshold'],'summary':summary},indent=2))

if __name__=='__main__':main()
