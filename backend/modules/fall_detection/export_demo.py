"""Export frozen TCS D validation scores to the existing website replay contract.

No fitting/inference/threshold selection. The UI state is only normal/suspected,
not a medically confirmed fall. Separate recording timelines are never stitched.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

from backend.shared.export import write_replay

MODULE=Path(__file__).resolve().parent
REPORT=MODULE/'reports/tcs_cross_subject_v1/comparison_seed42_v1/evaluation'
ARM='SA_CT_standardized_GCV'
DEFAULT_MEMBER='3/a_03_13_00.dat'


def build_replay(report=REPORT,member=DEFAULT_MEMBER):
    result=json.loads((report/(ARM+'.json')).read_text())
    if result['threshold']!=.5 or result['test_evaluated']:raise ValueError('Unexpected frozen validation protocol')
    checkpoint=MODULE/'outputs/tcs_cross_subject_v1/comparison_seed42_v1'/ARM/'best.pt'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=result['checkpoint_sha256']:raise ValueError('Frozen D checkpoint changed')
    grouped=defaultdict(list)
    source=report/(ARM+'_windows.csv')
    with source.open() as f:
        for row in csv.DictReader(f):
            if int(row['subject'])!=3 or row['role']!='validation':raise ValueError('Only validation03 replay permitted')
            score=float(row['fall_score']);end=float(row['window_end_seconds'])
            if not math.isfinite(score) or not 0<=score<=1:raise ValueError('Invalid score')
            grouped[row['member']].append((row,score,end))
    if len(grouped)!=12 or member not in grouped:raise ValueError('Expected all12 validation recordings')
    recordings=[]
    for name,group in sorted(grouped.items()):
        frames=[];previous=False;last=-1.
        for row,score,end in group:
            if end<=last:raise ValueError('Unordered recording')
            positive=score>=.5
            frames.append({'t':end,'fall_score':score,'state':'suspected' if positive else 'normal',
                'alarm_onset':positive and not previous,'threshold':.5,'recording_id':name,
                'window_start':float(row['window_start_seconds']),'window_end':end,
                'prediction':'Fall' if positive else 'Nonfall'})
            previous=positive;last=end
        annotation=float(group[0][0]['annotation_seconds'])
        recordings.append({'id':name,'title':name,'activity_id':int(group[0][0]['activity']),
            'annotation_seconds':annotation,'annotation_note':'Integer-second nominal event; onset/impact and time-zero not independently verified',
            'frames':frames,'background_alarm_episodes':sum(f['alarm_onset'] and f['t']<=120 for f in frames)})
    default=next(r for r in recordings if r['id']==member)
    sa=result['single_action'];ct=result['continuous']
    payload={'module':'fall_detection','version':'tcs-d-replay-v1','mock':False,'mode':'recorded_validation',
        'model_version':'tcs_cross_subject_SA_CT_standardized_GCV_seed42_v1','threshold':.5,
        'dataset':'TCS-Fall','subject':3,'split':'validation','default_recording_id':member,
        'checkpoint_sha256':result['checkpoint_sha256'],
        'provenance':{'kind':'real_recorded_validation','source':'Frozen D best epoch15; subject03 continuous validation',
            'source_csv_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'joint_ground_truth':False,'other_modules_share_recording':False},
        'notice':'Recorded validation replay. A positive score is suspected fall evidence; no new consecutive-window or notification rule. Other modules use independent data.',
        'frames':default['frames'],'recordings':recordings,
        'preprocessing':{'tiles':[{'label':'Dataset','value':'TCS-Fall / Intel5300'},
            {'label':'Input','value':'8000 packets → 1×400×90 GCV'},
            {'label':'Normalisation','value':'Training subjects01/02 only'},
            {'label':'Displayed split','value':'Validation subject03'}],
            'table':[{'stage':'Input','method':'Tx0, 3Rx×30 native positions; no padding'},
                {'stage':'Features','method':'Window-local Butterworth +20-packet GCV'},
                {'stage':'Scale','method':'Frozen per-feature train mean/std'}]},
        'training':{'table':[{'model':'D: five-layer 2D CNN','seed':42,'best_epoch':15,'epochs_run':25,'threshold':.5}]},
        'experiments':{'table':[
            {'model':'A: SA','background_alarms':21,'continuous_events':'12/12'},
            {'model':'B: SA + CT','background_alarms':17,'continuous_events':'12/12'},
            {'model':'C: SA + Z','background_alarms':28,'continuous_events':'12/12'},
            {'model':'D: SA + CT + Z','background_alarms':2,'continuous_events':'12/12'}]},
        'metrics':{'table':[
            {'scope':'SA validation','measure':'Fall precision','value':f"{sa['precision']:.2%}"},
            {'scope':'SA validation','measure':'Fall recall','value':f"{sa['recall']:.2%}"},
            {'scope':'SA validation','measure':'Fall F1','value':f"{sa['f1']:.2%}"},
            {'scope':'12 continuous validation trials','measure':'New-alert event hits','value':'12/12'},
            {'scope':'22.4min eligible background','measure':'False-alarm episodes','value':ct['background_false_alarm_episodes']},
            {'scope':'Short background rate estimate','measure':'Alarm episodes/hour','value':f"{ct['background_false_alarms_per_hour']:.2f}"},
            {'scope':'Nominal event annotation','measure':'Median alert endpoint offset','value':f"{ct['strict_detected_only_nominal_delay_median_seconds']:.2f}s (not end-to-end latency)"}],
            'confusion':{'labels':['Nonfall','Fall'],'matrix':[[sa['tn'],sa['fp']],[sa['fn'],sa['tp']]]}}}
    latency_file=MODULE/'reports/bc_integration_v1/frozen_D_cpu_inference.json'
    if latency_file.exists():
        measured=json.loads(latency_file.read_text())
        if measured['checkpoint_sha256']!=result['checkpoint_sha256']:raise ValueError('Latency used another model')
        latency=measured['latency'];payload['latency']=latency
        payload['metrics']['table'].extend([
            {'scope':'CPU batch1,2threads,30repeats','measure':'Model forward median /P95','value':f"{latency['forward_only']['median_ms']:.1f} / {latency['forward_only']['p95_ms']:.1f} ms"},
            {'scope':'Decode + feature + model in memory','measure':'Compute median /P95','value':f"{latency['memory_packet_decode_filter_gcv_normalize_forward']['median_ms']:.1f} / {latency['memory_packet_decode_filter_gcv_normalize_forward']['p95_ms']:.1f} ms; excludes packet wait and notification"}])
    return payload


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--bundle',action='store_true');args=ap.parse_args()
    payload=build_replay();write_replay('fall_detection',payload,bundle=args.bundle)
    print(json.dumps({'module':'fall_detection','recordings':len(payload['recordings']),'frames':sum(len(r['frames']) for r in payload['recordings']),
        'mock':False,'threshold':payload['threshold'],'default':payload['default_recording_id'],'bundled':args.bundle},indent=2))

if __name__=='__main__':main()
