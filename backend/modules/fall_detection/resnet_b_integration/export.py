"""Export existing CSI-Bench ResNet18 HP validation scores and B-reference slots."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np

from backend.shared.export import read_replay,write_replay
from backend.modules.fall_detection.activity_reference import segment_identity,attach_activity_reference

MODULE=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def build_payload():
    cfg=json.loads((HERE/'frozen_model.json').read_text());checkpoint=MODULE/cfg['checkpoint']
    if digest(checkpoint)!=cfg['checkpoint_sha256']:raise ValueError('ResNet18 checkpoint changed')
    selection=json.loads((MODULE/'reports/final_selection_v1_three_seeds/selection.json').read_text())
    run=next(r for r in selection['runs'] if r['kind']=='official_resnet18' and r['seed']==42)
    predictions=Path(run['prediction_file'])
    if digest(predictions)!=run['predictions_sha256']:raise ValueError('Saved validation predictions changed')
    with predictions.open() as f:pred=list(csv.DictReader(f))
    with (MODULE/'reports/sample_manifest.csv').open() as f:manifest={r['sample_id']:r for r in csv.DictReader(f)}
    frames=[];inputs=[]
    for row in pred:
        m=manifest[row['sample_id']]
        if m['split']!='val_id' or int(m['binary_label'])!=int(row['label']):raise ValueError('Validation label/split mismatch')
        if row['device']!='HP':continue
        with h5py.File(m['file_path'],'r') as h:raw=np.asarray(h['CSI_amps'])
        if raw.shape!=(232,500,1) or not np.isfinite(raw).all():raise ValueError('Unexpected HP raw input')
        # Identity is raw [feature,time] amplitudes, before either module normalizes.
        identity=segment_identity(raw[:,:,0]);score=float(row['fall_score'])
        if not np.isfinite(score) or not 0<=score<=1:raise ValueError('Invalid saved score')
        positive=score>=cfg['threshold'];truth=int(row['label'])
        frames.append({'t':float(len(frames)),'sample_index':len(frames),'sample_id':row['sample_id'],
            'recording_id':row['sample_id'],'input_id':identity,'device':'HP','fall_score':score,
            'threshold':cfg['threshold'],'state':'suspected' if positive else 'normal','prediction':'Fall' if positive else 'Nonfall',
            'ground_truth':'Fall' if truth else 'Nonfall','correct':int(positive)==truth,
            'model_version':cfg['version'],'time_basis':'independent segment presentation index; not physical recording time'})
        inputs.append({'sample_id':row['sample_id'],'file_path':m['file_path'],'input_id':identity,'device':'HP','binary_label':truth})
    if len(frames)!=926:raise ValueError('Expected all926 official HP validation records')
    tp=sum(f['prediction']=='Fall' and f['ground_truth']=='Fall' for f in frames)
    fp=sum(f['prediction']=='Fall' and f['ground_truth']=='Nonfall' for f in frames)
    fn=sum(f['prediction']=='Nonfall' and f['ground_truth']=='Fall' for f in frames)
    tn=len(frames)-tp-fp-fn;metrics={'n':len(frames),'tp':tp,'fp':fp,'fn':fn,'tn':tn,
        'precision':tp/(tp+fp),'recall':tp/(tp+fn),'f1':2*tp/(2*tp+fp+fn)}
    old=run['groups']['HP']['selected_global']
    if any(metrics[k]!=old[k] for k in ('n','tp','fp','fn','tn')):raise ValueError('Original HP metrics differ')
    payload={'module':'fall_detection','version':cfg['version'],'mock':False,'mode':'independent_segment_validation',
        'model_version':cfg['version'],'model_name':'CSI-Bench official ResNet18','dataset':'CSI-Bench / FallDetection',
        'split':'validation','device':'HP','threshold':cfg['threshold'],'threshold_origin':cfg['threshold_origin'],
        'checkpoint_sha256':cfg['checkpoint_sha256'],'label_mapping':cfg['label_mapping'],
        'timeline_kind':'presentation_only','physical_timestamps_available':False,
        'provenance':{'kind':'real_saved_validation_predictions','source_commit':cfg['source_commit'],'source_csv_sha256':digest(predictions),
            'joint_ground_truth':False,'other_modules_share_recording':False,'context_alignment':'exact raw amplitude input_id; no join on display time'},
        'frames':frames,'metrics_values':metrics,
        'preprocessing':{'tiles':[{'label':'Task / device','value':'CSI-Bench FallDetection / HP'},
            {'label':'Input','value':'[1,500,232] amplitude plane'},
            {'label':'Normalisation','value':'Official per-record mean/std'},
            {'label':'B reference','value':'Exact shared input identity required'}],
            'table':[{'step':'C input','method':'CSI_amps [232,500,1] → [1,500,232]'},
                {'step':'B input','method':'Same raw [232,500] amplitude; B keeps its own preprocessing'},
                {'step':'Reference','method':'B activity context never modifies the C fall score'}]},
        'training':{'table':[{'model':'Official ResNet18','source':'CSI-Bench trained weights','seed':42,'best_epoch':12,'integration_retraining':'none'}]},
        'experiments':{'table':[{'pipeline':'ResNet18 + B context','role':'Current CSI-Bench integration candidate'},
            {'pipeline':'Original 1D-CNN','role':'Preserved prior research-selection artifact'},
            {'pipeline':'TCS five-layer D CNN','role':'Separate archived continuous-data experiment; no transfer training'}]},
        'metrics':{'table':[{'scope':'926 HP validation segments','measure':'Fall Precision','value':f"{metrics['precision']:.2%}"},
            {'scope':'926 HP validation segments','measure':'Fall Recall','value':f"{metrics['recall']:.2%}"},
            {'scope':'926 HP validation segments','measure':'Fall F1','value':f"{metrics['f1']:.2%}"},
            {'scope':'Previously selected operating point','measure':'Global threshold','value':cfg['threshold']},
            {'scope':'B/C reference','measure':'Joint accuracy','value':'Not evaluated; paired activity ground truth unavailable'},
            {'scope':'Independent segments','measure':'Event recall / false alarms per hour','value':'Not defined for this replay'}],
            'confusion':{'labels':['Nonfall','Fall'],'matrix':[[tn,fp],[fn,tp]]}}}
    latency_path=MODULE/'reports/hp_latency_v1/official_resnet18_cpu.json'
    if latency_path.exists():
        measured=json.loads(latency_path.read_text())
        if measured['checkpoint_sha256']!=cfg['checkpoint_sha256']:raise ValueError('Latency belongs to another checkpoint')
        timing={x['scope']:x for x in measured['scopes']}
        payload['latency']={'source':'Previous HP CPU timing on the exact same checkpoint','measurement':measured,'includes_module_B':False}
        payload['metrics']['table'].extend([
            {'scope':'C only, previous CPU batch1 timing','measure':'Forward median /P95','value':f"{timing['forward_only']['median_ms']:.1f} / {timing['forward_only']['p95_ms']:.1f} ms"},
            {'scope':'C only, previous in-memory pipeline','measure':'Preprocess + model median /P95','value':f"{timing['memory_pipeline']['median_ms']:.1f} / {timing['memory_pipeline']['p95_ms']:.1f} ms"}])
    report=MODULE/'reports/resnet_b_integration_v1';report.mkdir(parents=True,exist_ok=True)
    (report/'inputs_manifest.json').write_text(json.dumps(inputs,indent=2))
    (report/'export_receipt.json').write_text(json.dumps({'checkpoint_sha256':cfg['checkpoint_sha256'],'predictions_sha256':digest(predictions),'threshold':cfg['threshold'],
        'records':len(frames),'metrics':metrics,'test_inference':False,'retraining':False,'threshold_retuned':False,'labels_from_original_manifest':True},indent=2))
    return attach_activity_reference(payload,read_replay('human_activity_recognition'))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--bundle',action='store_true');args=ap.parse_args()
    p=build_payload();write_replay('fall_detection',p,bundle=args.bundle)
    print(json.dumps({'model':p['model_name'],'records':len(p['frames']),'threshold':p['threshold'],'metrics':p['metrics_values'],'B_reference':p['activity_reference_summary']},indent=2))

if __name__=='__main__':main()
