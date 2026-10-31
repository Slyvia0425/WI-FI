"""Compare saved validation predictions only; never loads or predicts test records."""
import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def metrics(records,threshold):
    y=np.array([r['label'] for r in records],dtype=int)
    p=np.array([r['fall_score'] for r in records])>=threshold
    tp=int(((y==1)&p).sum());fp=int(((y==0)&p).sum())
    fn=int(((y==1)&~p).sum());tn=int(((y==0)&~p).sum())
    return {'n':len(y),'fall':int(y.sum()),'nonfall':int((y==0).sum()),'tp':tp,'fp':fp,'fn':fn,'tn':tn,
            'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,
            'fall_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            'false_positive_rate':fp/(fp+tn) if fp+tn else 0.}


def select_threshold(records):
    scores=np.array([r['fall_score'] for r in records],dtype=np.float64)
    labels=np.array([r['label'] for r in records],dtype=int)
    if not np.isfinite(scores).all() or np.any((scores<0)|(scores>1)) or set(labels)!={0,1}:
        raise ValueError('Both verified labels and finite [0,1] Fall scores required')
    unique=np.unique(scores)
    candidates=[0.]+((unique[:-1]+unique[1:])/2).tolist()
    rows=[{'threshold':float(t),**metrics(records,t)} for t in candidates]
    def key(r):return (Fraction(2*r['tp'],2*r['tp']+r['fp']+r['fn']),Fraction(r['tp'],r['fall']),r['threshold'])
    return max(rows,key=key),rows


def load_validation(path,expected):
    rows=list(csv.DictReader(path.open()))
    result=[{'sample_id':r['sample_id'],'device':r['device'],'label':int(r['label']),'fall_score':float(r['fall_score'])} for r in rows]
    if len(result)!=len(expected) or len({r['sample_id'] for r in result})!=len(expected):raise ValueError('Missing/duplicate validation IDs')
    if {r['sample_id'] for r in result}!=set(expected):raise ValueError('Input is not the exact official val_id set')
    for r in result:
        e=expected[r['sample_id']]
        if r['label']!=int(e['binary_label']) or r['device']!=e['device']:raise ValueError('Label or device mismatch')
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--candidate',action='append',required=True,help='name=/absolute/path/to/validation.csv')
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args()
    if args.output_dir.exists():raise FileExistsError('Choose a fresh report directory')
    manifest=list(csv.DictReader(args.manifest.open()))
    expected={r['sample_id']:r for r in manifest if 'val_id' in r['split'].split(';')}
    assert len(expected)==1005
    args.output_dir.mkdir(parents=True)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    results={};flat=[]
    for item in args.candidate:
        name,filename=item.split('=',1);path=Path(filename)
        if name in results:raise ValueError('Duplicate candidate name')
        records=load_validation(path,expected);selected,candidates=select_threshold(records)
        with (args.output_dir/f'{name}_threshold_candidates.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(candidates[0]));w.writeheader();w.writerows(candidates)
        groups={'all':records,**{d:[r for r in records if r['device']==d] for d in ('HP','ESP32')}}
        result={'prediction_file':str(path.resolve()),'prediction_sha256':sha(path),'selected':selected,'groups':{}}
        for (group,subset),ax in zip(groups.items(),axes):
            y=[r['label'] for r in subset];scores=[r['fall_score'] for r in subset]
            precision,recall,_=precision_recall_curve(y,scores)
            ap_score=float(average_precision_score(y,scores))
            curve,=ax.step(recall,precision,where='post',label=f'{name} AP={ap_score:.3f}')
            result['groups'][group]={'average_precision':ap_score}
            for operating,t in [('fixed_0_5',.5),('selected_global',selected['threshold'])]:
                m=metrics(subset,t);result['groups'][group][operating]={'threshold':t,**m}
                flat.append({'model':name,'group':group,'operating_point':operating,'threshold':t,**m})
                ax.scatter(m['recall'],m['precision'],marker='o' if operating=='fixed_0_5' else 'x',s=25,color=curve.get_color())
        results[name]=result
    for group,ax in zip(('all','HP','ESP32'),axes):
        ax.set(title=group+' validation',xlabel='Fall Recall',ylabel='Fall Precision',xlim=(0,1.02),ylim=(0,1.02));ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Same official val_id; one global threshold per candidate; no test evaluation\nCircles: threshold 0.5; crosses: validation-selected threshold')
    fig.savefig(args.output_dir/'precision_recall_comparison.png',dpi=160);plt.close(fig)
    with (args.output_dir/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    report={'scope':'validation only; exploratory model and threshold selection, not unbiased final performance',
            'manifest_sha256':sha(args.manifest),'objective':'maximum overall Fall F1; higher Recall then higher threshold',
            'prediction_rule':'fall_score >= threshold','device_specific_thresholds':False,'test_evaluated':False,'models':results}
    (args.output_dir/'comparison.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
