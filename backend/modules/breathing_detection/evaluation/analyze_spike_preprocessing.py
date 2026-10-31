"""Descriptive spike-source findings and fixed-candidate plots; never tune rules."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.io import loadmat

from ..preprocessing.isolated_spikes import isolated_spike_filter,preprocessing_configs
from ..preprocessing.respiratory_rate import spectra
from ..preprocessing.wire_breath import WireBreathDataset
from .evaluate_spike_preprocessing import raw_slice,candidate_window

ROOT=Path('backend/modules/breathing_detection/outputs/spike_preprocessing_v1')


def plot_case(row,ref,sample,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    raw=loadmat(ref.path,squeeze_me=True,variable_names=('time','csi_abs'))
    t=np.asarray(raw['time'],dtype=float).reshape(-1)
    values=np.asarray(raw['csi_abs'],dtype=np.float32)
    targets=ref.start_seconds+np.arange(900)/30
    rt,rx=raw_slice(t,values,targets)
    raw_clean={h:isolated_spike_filter(rt,rx,h)[0] for h in (6.,8.)}
    processed={c['name']:candidate_window(t,values,targets,sample.x,c)[0] for c in preprocessing_configs()}
    mean=rx.mean(axis=1)
    center,scale=float(mean.mean()),max(float(mean.std()),1e-8)
    normalize=lambda v:(np.mean(v,axis=1)-center)/scale
    styles={'baseline':('#555555','-'),'raw_hampel6':('#146b96','-'),'raw_hampel8':('#e08620','--'),
            'resampled_hampel6':('#19845a','-'),'resampled_hampel8':('#9b3b79','--')}
    fig,axes=plt.subplots(3,2,figsize=(14,9),layout='constrained')
    peak_time=float(rt[int(np.argmax(mean))]-ref.start_seconds)
    for j,zoom in enumerate((False,True)):
        r=0 if not zoom else 2
        axes[r,0].plot(rt-ref.start_seconds,normalize(rx),color='#555555',label='Raw source',linewidth=.6)
        for h in (6.,8.):
            color,style=styles[f'raw_hampel{int(h)}']
            axes[r,0].plot(rt-ref.start_seconds,normalize(raw_clean[h]),color=color,linestyle=style,label=f'Raw clean {int(h)}',linewidth=.8)
        axes[r,1].plot(targets-ref.start_seconds,normalize(sample.x[0].T),color='#555555',label='Current 30Hz',linewidth=.6)
        for h in (6.,8.):
            color,style=styles[f'resampled_hampel{int(h)}']
            axes[r,1].plot(targets-ref.start_seconds,normalize(processed[f'resampled_hampel{int(h)}'][0].T),color=color,linestyle=style,label=f'Post clean {int(h)}',linewidth=.8)
        for col in (0,1):
            axes[r,col].set(xlabel='Time (s)',ylabel='Mean CSI, source-window scale',xlim=(max(0,peak_time-.3),min(30,peak_time+.3)) if zoom else (0,30))
            axes[r,col].legend(fontsize=7)
        axes[r,0].set_title('Native sampling'+(' — peak zoom' if zoom else ' — full 30s'),fontsize=10)
        axes[r,1].set_title('Uniform 30Hz'+(' — same time zoom' if zoom else ' — full 30s'),fontsize=10)
    for name,x in {'baseline':sample.x,**processed}.items():
        f,power=spectra(x[0],band_hz=(.05,5.))
        mean_power=np.maximum(power.mean(axis=0),1e-12)
        color,style=styles[name]
        rate=row['predictions'][name]
        label=f"{name}: {rate:.2f}" if rate is not None else f'{name}: unavailable'
        for ax in axes[1]:
            ax.semilogy(f*60,mean_power,color=color,linestyle=style,label=label,linewidth=.9)
    for col in (0,1):
        axes[1,col].axvline(row['reference_bpm'],color='black',linestyle=':',label='Reference (evaluation only)')
        axes[1,col].set(xlabel='Frequency (cycles/min)',ylabel='Mean normalized power',xlim=(3,60) if col==0 else (3,300))
        axes[1,col].legend(fontsize=6)
    axes[1,0].set_title('Respiratory-band context; legend rates in BPM',fontsize=10)
    axes[1,1].set_title('Wider spectrum; each method uses existing normalization',fontsize=10)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle(f"Spike preprocessing — P{row['participant_id']} / {row['posture']} / {row['environment']}\n"
                 f"baseline error {row['baseline_error']:.2f} BPM; spikes are not physical-motion labels",fontsize=12)
    fig.savefig(path,dpi=120,bbox_inches='tight')
    plt.close(fig)


def main():
    target=ROOT/'analysis.json'
    if target.exists():
        raise FileExistsError('Preserve spike descriptive analysis')
    audits=json.loads((ROOT/'raw_audit.json').read_text())
    rows=json.loads((ROOT/'window_predictions.json').read_text())
    grouped={}
    for audit in audits:
        grouped.setdefault(audit['peak_frame_hash'],[]).append(audit)
    signatures=[]
    for signature,group in sorted(grouped.items(),key=lambda item:-len(item[1]))[:10]:
        if len(group)<2:
            continue
        signatures.append({'hash':signature,'files':len(group),'recordings':len({a['recording_id'] for a in group}),
                           'persons':sorted({a['metadata']['participant_id'] for a in group}),
                           'rooms':sorted({a['metadata']['room'] for a in group}),
                           'median_occurrences_per_file':float(np.median([a['peak_frame_occurrences_in_file'] for a in group]))})
    by_id={r['sample_id']:r for r in rows}
    error_groups={}
    for name,predicate in [('accurate',lambda e:e<=2),('intermediate',lambda e:2<e<=5),('large',lambda e:e>5)]:
        members=[a for a in audits if a['sample_id'] in by_id and predicate(by_id[a['sample_id']]['baseline_error'])]
        error_groups[name]={'windows':len(members),
            'median_raw_excursion_frames':float(np.median([a['raw_excursion_frames'] for a in members])),
            'median_resampled_excursion_frames':float(np.median([a['resampled_excursion_frames'] for a in members])),
            'median_interval_before_excursion':float(np.median([a['interval_before_excursion_median'] for a in members if a['interval_before_excursion_median'] is not None]))}
    edits={}
    for c in preprocessing_configs():
        available=[r['edits'][c['name']] for r in rows if r['edits'][c['name']]['available']]
        edits[c['name']]={'available':len(available),
             'median_replaced_fraction':float(np.median([e['replaced_fraction'] for e in available])),
             'max_replaced_fraction':max(e['replaced_fraction'] for e in available),
             'median_replaced_frames':float(np.median([e['replaced_frames'] for e in available]))}
    out={'purpose':'descriptive source audit only; no physical-cause attribution or parameter tuning',
         'extreme_frame_signatures':signatures,'by_baseline_error':error_groups,'edits':edits,
         'median_interval_before_excursion':float(np.median([a['interval_before_excursion_median'] for a in audits if a['interval_before_excursion_median'] is not None])),
         'median_interval_before_nonexcursion':float(np.median([a['interval_before_nonexcursion_median'] for a in audits if a['interval_before_nonexcursion_median'] is not None])),
         'all_start_offsets':[float(v) for v in sorted({a['start_offset_from_first_csi'] for a in audits})],
         'plots':[]}
    selected=sorted([r for r in rows if r['baseline_error']>5],key=lambda r:(-r['baseline_error'],r['sample_id']))[:3]
    ids={r['sample_id'] for r in selected}
    dataset=WireBreathDataset('data/WiReBreath',window_seconds=30,stride_seconds=30,normalize=False)
    samples={ref.sample_id:(ref,dataset[i]) for i,ref in enumerate(dataset._windows) if ref.sample_id in ids}
    for i,row in enumerate(selected,1):
        path=ROOT/f'preprocessing_case_{i:02d}.png'
        if path.exists():
            raise FileExistsError('Preserve plot')
        ref,sample=samples[row['sample_id']]
        plot_case(row,ref,sample,path)
        out['plots'].append({'path':str(path),'sample_id':row['sample_id']})
    target.write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    main()
