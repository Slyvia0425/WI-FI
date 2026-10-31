"""Descriptive matched-error analysis after fixed pollution-rule assessment."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.signal import detrend

from ..preprocessing.motion_pollution import FEATURES
from ..preprocessing.respiratory_rate import spectra
from ..preprocessing.wire_breath import WireBreathDataset

ROOT = Path('backend/modules/breathing_detection/outputs/motion_pollution_v1')


def rate_error(row):
    return abs(row['predictions']['unified']-row['reference_bpm'])


def matched_controls(target,rows):
    good = sorted([r for r in rows if rate_error(r)<=2 and r['sample_id']!=target['sample_id']],key=lambda r:r['sample_id'])
    tiers = [('same_recording',('recording_id',)),
             ('same_person_posture_room_environment',('participant_id','posture','room','environment')),
             ('same_person_posture_room',('participant_id','posture','room'))]
    for tier,keys in tiers:
        selected = [r for r in good if all(r[k]==target[k] for k in keys)]
        if selected:
            return tier,selected[:3]
    return 'unmatched',[]


def describe(rows):
    valid = [r for r in rows if r['features']['available']]
    return {'windows':len(rows),'recordings':len({r['recording_id'] for r in rows}),
            'features':{key:{'q25':float(np.quantile([r['features'][key] for r in valid],.25)),
                             'median':float(np.median([r['features'][key] for r in valid])),
                             'q75':float(np.quantile([r['features'][key] for r in valid],.75))} if valid else None for key in FEATURES}}


def plot_case(target,control,samples,path,number):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,3,figsize=(15,6),layout='constrained')
    for j,row in enumerate((target,control)):
        sample = samples[row['sample_id']]
        x = np.asarray(sample.x[0],dtype=float)
        centered = detrend(x,axis=1)
        z = centered/np.maximum(centered.std(axis=1,keepdims=True),1e-8)
        axes[j,0].plot(np.arange(900)/30,z.mean(axis=0),color='#167a79',linewidth=.8)
        axes[j,0].set(xlabel='Time (s)',ylabel='Mean standardized CSI',xlim=(0,30))
        blocks = np.sqrt(np.mean(z.reshape(-1,30,30)**2,axis=2))
        energy = np.median(blocks/np.maximum(np.median(blocks,axis=1,keepdims=True),1e-6),axis=0)
        axes[j,1].plot(np.arange(30)+.5,energy,color='#8b4d1d',marker='.',linewidth=1)
        axes[j,1].set(xlabel='Time (s)',ylabel='Median relative 1s RMS',xlim=(0,30))
        f,power = spectra(x,band_hz=(.1,5.))
        axes[j,2].semilogy(f*60,np.maximum(power.mean(axis=0),1e-12),color='#3e568e',linewidth=1)
        axes[j,2].axvspan(6,42,color='#e5bd6b',alpha=.2,label='Rate search band')
        axes[j,2].axvline(row['predictions']['unified'],color='#b94e2e',label='Fixed estimator')
        axes[j,2].axvline(row['reference_bpm'],color='#228746',linestyle='--',label='Reference (evaluation only)')
        axes[j,2].set(xlabel='Frequency (cycles/min)',ylabel='Mean normalized power',xlim=(6,300))
        axes[j,2].legend(fontsize=7,loc='upper right')
        for ax in axes[j]:
            ax.grid(alpha=.2)
        role = 'Large-error target' if j==0 else 'Matched accurate control'
        axes[j,0].set_title(f"{role}\nP{row['participant_id']} / {row['posture']} / {row['environment']} / error {rate_error(row):.2f} BPM",fontsize=9)
        axes[j,1].set_title(f"jump={row['features']['jump_ratio']:.2f}; burst={row['features']['energy_burst_ratio']:.2f}",fontsize=10)
        axes[j,2].set_title(f"broadband={row['features']['broadband_fraction']:.3f}; incoherence={row['features']['adjacent_incoherence']:.3f}",fontsize=10)
    fig.suptitle(f'Case {number}: CSI pollution diagnostics, not physical-motion ground truth',fontsize=13)
    fig.savefig(path,dpi=130,bbox_inches='tight')
    plt.close(fig)


def main():
    target = ROOT/'diagnostics.json'
    if target.exists():
        raise FileExistsError('Preserve descriptive diagnostics')
    rows = json.loads((ROOT/'window_results.json').read_text())
    large = sorted([r for r in rows if rate_error(r)>5],key=lambda r:(-rate_error(r),r['sample_id']))
    cases = []
    for row in large:
        tier,controls = matched_controls(row,rows)
        valid = row['features']['available'] and all(c['features']['available'] for c in controls)
        cases.append({'sample_id':row['sample_id'],'recording_id':row['recording_id'],'participant_id':row['participant_id'],
            'posture':row['posture'],'room':row['room'],'environment':row['environment'],
            'candidate_bpm':row['predictions']['unified'],'reference_bpm':row['reference_bpm'],'error_bpm':rate_error(row),
            'features':row['features'],'matching_tier':tier,'control_ids':[c['sample_id'] for c in controls],
            'environment_mismatch':any(c['environment']!=row['environment'] for c in controls),
            'target_minus_control_median':{key:row['features'][key]-float(np.median([c['features'][key] for c in controls]))
                                          for key in FEATURES} if controls and valid else None})
    paired = [c for c in cases if c['target_minus_control_median'] is not None]
    report = {'purpose':'post-run descriptive comparison only; no rule/threshold fitting or motion labels',
              'by_error':{'accurate_le2':describe([r for r in rows if rate_error(r)<=2]),
                          'intermediate_2to5':describe([r for r in rows if 2<rate_error(r)<=5]),
                          'large_gt5':describe(large)},
              'by_environment':{env:describe([r for r in rows if r['environment']==env]) for env in ('Quiet','Interference')},
              'matching_tiers':dict(Counter(c['matching_tier'] for c in cases)),
              'environment_mismatch_cases':sum(c['environment_mismatch'] for c in cases),
              'unique_controls':len({i for c in cases for i in c['control_ids']}),
              'matched_comparisons':{key:{'cases':len(paired),
                   'target_above_control_median':sum(c['target_minus_control_median'][key]>0 for c in paired),
                   'median_paired_difference':float(np.median([c['target_minus_control_median'][key] for c in paired])) if paired else None} for key in FEATURES},
              'large_error_cases':cases,'plots':[]}
    selected = [case for case in cases if case['control_ids']][:3]
    ids = {case['sample_id'] for case in selected}|{case['control_ids'][0] for case in selected}
    dataset = WireBreathDataset('data/WiReBreath',window_seconds=30,stride_seconds=30,normalize=False)
    samples = {ref.sample_id:dataset[i] for i,ref in enumerate(dataset._windows) if ref.sample_id in ids}
    by_id = {r['sample_id']:r for r in rows}
    for i,case in enumerate(selected,1):
        path = ROOT/f'matched_case_{i:02d}.png'
        if path.exists():
            raise FileExistsError('Preserve plot artifact')
        plot_case(by_id[case['sample_id']],by_id[case['control_ids'][0]],samples,path,i)
        report['plots'].append({'path':str(path),'target':case['sample_id'],'control':case['control_ids'][0],'tier':case['matching_tier']})
    target.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('large_error_cases','plots')},indent=2))
    print('Plots:',report['plots'])


if __name__ == '__main__':
    main()
