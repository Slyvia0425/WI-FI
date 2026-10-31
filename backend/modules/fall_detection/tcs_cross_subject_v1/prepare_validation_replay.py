"""Prepare only subject03 continuous validation replay, with fixed event metrics."""
import argparse
import json
from pathlib import Path
import sys
import time
import zipfile
import numpy as np
from scipy import signal

HERE=Path(__file__).resolve().parent
MODULE=HERE.parent
sys.path.insert(0,str(HERE))
from prepare_features import audited_bytes, decode_amplitude, make_feature, sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--eligibility',type=Path,required=True);ap.add_argument('--features',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise FileExistsError('Preserve earlier replay preparation')
    info=json.loads((args.eligibility/'eligibility.json').read_text())
    records=json.loads((args.eligibility/'records.json').read_text())
    selected=sorted([r for r in records if r['input_eligibility'].startswith('eligible') and r['role']=='validation'
                     and r['source_kind']=='annotated_continuous_candidate'],key=lambda r:r['archive_member'])
    if len(selected)!=12 or {r['nominal_subject_id'] for r in selected}!={3}:raise ValueError('Unexpected replay subjects')
    archive_info=next(a for a in info['archives'] if a['subject']==3)
    if sha(Path(archive_info['path']))!=archive_info['sha256']:raise ValueError('Validation archive changed')
    args.output.mkdir(parents=True)
    protocol={'subject':3,'role':'validation','threshold':.5,'window_packets':8000,'hop_packets':1000,
              'decision_time':'actual NIC timestamp of the last packet in each trailing window, relative to first packet',
              'filter':'same window-local filtfilt/GCV as training; no packets after each window endpoint',
              'alarm':'one new alarm at each negative-to-positive transition; first output counts if positive; no smoothing, cooldown or extra threshold',
              'strict_event_match':'new alarm onset in [nominal annotation, annotation+11s]',
              'tolerant_event_match':'new alarm onset in [annotation-3s, annotation+11s]',
              'background':'documented initial120s only; exposure starts at first complete-window endpoint and ends at120s',
              'false_alarm_rate':'new alarm onsets in background / eligible background hours',
              'positive_state_fraction':'hold each prediction until next endpoint, clip to eligible initial120s, sum positive seconds / exposure',
              'paper_proxy':'positive window interval overlaps annotation+-3s for CWR; absence of positive windows outside this overlap for NEWR',
              'latency':'nominal annotation-to-new-alarm endpoint offset; onset/impact/time-zero unverified; compute and notification latency not measured',
              'evaluation':'same frozen protocol for all four arms; no threshold search or test subject04 evaluation',
              'full_continuous_scores_not_yet_computed':True}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    base=json.loads((args.features/'preparation.json').read_text())
    base_rows=json.loads((args.features/'samples.json').read_text())
    lookup={r['sample_id']:i for i,r in enumerate(base_rows) if r['role']=='validation' and r['kind']=='continuous_background'}
    cache=np.memmap(args.features/'features.f32',dtype=np.float32,mode='r',shape=tuple(base['shape']))
    b,a=signal.butter(8,.2,'lowpass');rows=[];comparisons=0;maximum=0.;begin=time.perf_counter()
    with zipfile.ZipFile(archive_info['path']) as z,(args.output/'features.f32').open('wb') as out:
        for r in selected:
            data=audited_bytes(z,r);amp,elapsed=decode_amplitude(data)
            for window_index,start in enumerate(range(0,len(amp)-8000+1,1000)):
                feature=make_feature(amp[start:start+8000],b,a)
                sample_id=r['archive_member']+f':packet{start}'
                if sample_id in lookup:
                    difference=float(np.max(np.abs(feature-cache[lookup[sample_id]])))
                    maximum=max(maximum,difference);comparisons+=1
                    if difference>1e-7:raise ValueError('Replay differs from existing validation background features')
                out.write(np.ascontiguousarray(feature).tobytes())
                rows.append({'member':r['archive_member'],'subject':3,'role':'validation','activity':r['activity_id'],'repeat':r['repeat_id'],
                             'sample_id':sample_id,'window_index':window_index,'start_packet':start,'end_packet':start+7999,
                             'window_start_seconds':float(elapsed[start]),'window_end_seconds':float(elapsed[start+7999]),
                             'annotation_seconds':r['annotation_record']['recorded_fall_time_seconds']})
            print(json.dumps({'prepared_validation_continuous':r['archive_member'],'windows_total':len(rows),'elapsed_seconds':time.perf_counter()-begin}),flush=True)
            del amp,elapsed,data
    if comparisons!=len(lookup):raise ValueError('Not all background windows matched replay')
    (args.output/'windows.json').write_text(json.dumps(rows,indent=2))
    summary={'shape':[len(rows),1,400,90],'feature_cache_sha256':sha(args.output/'features.f32'),
             'windows_sha256':sha(args.output/'windows.json'),'protocol_sha256':sha(args.output/'protocol.json'),
             'background_windows_compared':comparisons,'max_abs_difference_to_common_validation_features':maximum,
             'source_archive_sha256':archive_info['sha256'],'test_subject_read':False,'model_predictions_run':False}
    (args.output/'preparation.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
