"""Describe exact within-record source overlap, without changing evaluation splits."""
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.io import loadmat

ROOT=Path('backend/modules/breathing_detection/outputs/spike_preprocessing_v1')


def prefix_suffix_overlap(a,b):
    candidates=np.flatnonzero(np.all(a==b[0],axis=1))
    best=(0,None)
    for offset in candidates:
        count=min(len(a)-int(offset),len(b))
        if count>best[0] and np.array_equal(a[offset:offset+count],b[:count]):
            best=(count,int(offset))
    return best


def main():
    target=ROOT/'source_overlap_audit.json'
    if target.exists():
        raise FileExistsError('Preserve overlap audit')
    audits=json.loads((ROOT/'raw_audit.json').read_text())
    groups=defaultdict(list)
    for row in audits:
        groups[row['recording_id']].append(row)
    pairs=[]
    extreme_complex=defaultdict(set)
    for recording,group in sorted(groups.items()):
        previous=None
        for row in sorted(group,key=lambda r:r['metadata']['segment_id']):
            path=Path(row['source_path'])
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row['raw_sha256']:
                raise ValueError('Raw source changed since main audit')
            raw=loadmat(path,squeeze_me=True,variable_names=('time','csi_complex','csi_abs'))
            complex_values=np.asarray(raw['csi_complex'])
            t=np.asarray(raw['time'],dtype=float).reshape(-1)
            peak=int(np.argmax(np.asarray(raw['csi_abs']).mean(axis=1)))
            extreme_complex[row['peak_frame_hash']].add(hashlib.sha256(complex_values[peak].tobytes()).hexdigest())
            if previous is not None:
                prev_row,a,ta=previous
                count,offset=prefix_suffix_overlap(a,complex_values)
                aligned=bool(np.allclose(ta[offset:offset+count]-ta[offset],t[:count]-t[0],atol=1e-4,rtol=0)) if count else False
                pairs.append({'recording_id':recording,'previous':prev_row['sample_id'],'next':row['sample_id'],
                    'segment_step':int(row['metadata']['segment_id']-prev_row['metadata']['segment_id']),
                    'exact_complex_overlap_points':count,'previous_start_index':offset,'relative_time_matches':aligned,
                    'previous_stride_seconds':float(ta[offset]-ta[0]) if count else None})
            previous=(row,complex_values,t)
    report={'purpose':'post-audit descriptive source-overlap check; no new preprocessing/rule selection',
            'recordings':len(groups),'adjacent_segment_pairs':len(pairs),
            'overlapping_pairs':sum(p['exact_complex_overlap_points']>0 for p in pairs),
            'overlap_point_counts':dict(Counter(p['exact_complex_overlap_points'] for p in pairs)),
            'relative_time_matched_overlap_pairs':sum(p['relative_time_matches'] for p in pairs),
            'all_shared_extremes_within_one_recording':all(len({a['recording_id'] for a in audits if a['peak_frame_hash']==signature})==1
                for signature,hashes in extreme_complex.items() if sum(a['peak_frame_hash']==signature for a in audits)>1),
            'shared_extreme_magnitude_complex_variants':[{'magnitude_hash':signature,'complex_variants':len(hashes)}
                for signature,hashes in extreme_complex.items() if sum(a['peak_frame_hash']==signature for a in audits)>1],
            'pairs':pairs,
            'interpretation':'Exact repeated source samples mean these files are not statistically independent. Whole-record grouping was already enforced, so do not change splits posthoc. Shared magnitude alone is not proof of clipping/AGC/fabrication.'}
    target.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('pairs','shared_extreme_magnitude_complex_variants')},indent=2))


if __name__=='__main__':
    main()
