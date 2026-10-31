"""Render saved epoch histories without interacting with the training process."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--report',type=Path,required=True);args=ap.parse_args()
    names=['SA_raw_GCV','SA_CT_raw_GCV','SA_standardized_GCV','SA_CT_standardized_GCV']
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,name in zip(axes.flat,names):
        path=args.report/name/'history.csv'
        if not path.exists():ax.text(.5,.5,'Not started',ha='center',va='center',transform=ax.transAxes);ax.set_title(name);continue
        with path.open() as f:rows=list(csv.DictReader(f))
        rows=[r for r in rows if r.get('epoch') and r.get('val_loss')]
        x=[int(r['epoch']) for r in rows];train=[max(float(r['train_loss']),1e-10) for r in rows];val=[float(r['val_loss']) for r in rows]
        ax.plot(x,train,label='training CE');ax.plot(x,val,label='fixed validation CE');ax.set_yscale('log')
        done=(args.report/name/'summary.json').exists();ax.set_title(name+f' | {len(rows)} epochs'+(' | complete' if done else ' | running'))
        ax.set_xlabel('Epoch');ax.set_ylabel('Loss (log scale)');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Saved training progress | seed42 | subject04 not evaluated')
    fig.savefig(args.report/'progress_curves.png',dpi=150);plt.close(fig)

if __name__=='__main__':main()
