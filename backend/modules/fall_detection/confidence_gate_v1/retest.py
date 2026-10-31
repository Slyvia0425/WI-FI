"""Fresh-process frozen-model gate retest:926 HP val +25 HAR activity examples.
No predictions are read from replay for execution; historical outputs are only
compared after each new prediction. No model or production replay is changed.
"""
import csv,json,hashlib,sys,time
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import torch
from backend.modules.fall_detection.final_selection_v1.runtime import read_amplitude,load_candidate,predict_files
from backend.modules.fall_detection.activity_reference import segment_identity,activity_reference
from backend.modules.human_activity_recognition.integration_v1.runtime import ReleasedActivityModel
from .policy import execute,config
M=Path(__file__).resolve().parents[1];B=M.parent/'human_activity_recognition';OUT=M/'reports/confidence_gate_retest_v1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 if OUT.exists():raise FileExistsError('Preserve this run; choose a separate version')
 OUT.mkdir()
 cfg=json.loads((M/'resnet_b_integration/frozen_model.json').read_text())
 bcfg=json.loads((B/'integration_v1/frozen_model.json').read_text())
 protected=[M/cfg['checkpoint'],B/bcfg['checkpoint'],M/'resnet_b_integration/frozen_model.json',B/'integration_v1/frozen_model.json',M/'confidence_gate_v1/policy.json',M/'outputs/replay.json',B/'outputs/replay.json']
 before={str(p):sha(p) for p in protected}
 manifest=list(csv.DictReader((M/'reports/sample_manifest.csv').open()))
 archived=json.loads((M/'outputs/before_confidence_gate_v1.json').read_text());byid={r['sample_id']:r for r in manifest}
 samples=[]
 for f in archived['frames']:
  r=byid[f['sample_id']];assert r['split']=='val_id' and r['device']=='HP'
  samples.append({'cohort':'HP_FallDetection_val','sample_id':r['sample_id'],'file_path':r['file_path'],'label':r['original_label'],'reference_input_id':f['input_id'],'reference_B_confidence':f['activity_reference']['confidence'],'reference_C_score':f['fall_score'],'device':'HP','binary_label':int(r['binary_label'])})
 for r in json.loads((M/'reports/activity_probe_v1/records.json').read_text()):
  assert r['status']=='ok'
  samples.append({'cohort':'HAR_activity_probe','sample_id':r['sample_id'],'file_path':r['source']['local_path'],'label':r['source']['label'],'reference_input_id':r['raw_identity_C'],'reference_B_confidence':r['B_prediction']['confidence'],'reference_C_score':r['C_fall_score'],'device':'HP','binary_label':0})
 assert len(samples)==951
 (OUT/'protocol.json').write_text(json.dumps({'n':951,'cohorts':{'HP_FallDetection_val':926,'HAR_activity_probe':25},'policy':config(),'seed42_models':True,'training':False,'C_threshold':cfg['threshold'],'method':'New Python process; B actual prediction batched32; lazy C model loading; C callback invoked per selected input only. Historical scores only used for agreement checks. No independent test/generalization claim.'},indent=2))
 start=time.perf_counter();torch.set_num_threads(2);bmodel=ReleasedActivityModel();cmodel=None;cstate=None;c_loads=0;c_calls=[];records=[]
 for offset in range(0,len(samples),32):
  group=samples[offset:offset+32];arrays=[read_amplitude(r['file_path']) for r in group]
  assert all(a.shape==(232,500) and np.isfinite(a).all() for a in arrays)
  bp=bmodel.predict_many(arrays)
  for row,a,bout in zip(group,arrays,bp):
   identity=segment_identity(a);assert identity==row['reference_input_id']
   ref=activity_reference({'input_id':identity,'device':'HP'},{'mock':False,'frames':[{**bout,'input_id':identity,'device':'HP'}]})
   def c_predict():
    nonlocal cmodel,cstate,c_loads
    if cmodel is None:
     assert sha(M/cfg['checkpoint'])==cfg['checkpoint_sha256']
     cmodel,cstate=load_candidate(M/cfg['checkpoint'],'official_resnet18');c_loads+=1
    c_calls.append(row['sample_id'])
    score=predict_files(cmodel,cstate,'official_resnet18',[row])[0]['fall_score'];positive=score>=cfg['threshold']
    return {'fall_score':score,'prediction':'Fall' if positive else 'Nonfall','is_fall':positive,'state':'suspected' if positive else 'normal'}
   calls_before=len(c_calls);result=execute(ref,c_predict)
   assert len(c_calls)-calls_before==int(result['C_ran'])
   if not result['C_ran']:assert result['fall_score'] is None and result['prediction'] is None
   r={**row,'input_id':identity,'B_confidence':bout['confidence'],'B_activity':bout['activity'],'B_quality':bout['quality'],'B_confidence_abs_difference':abs(bout['confidence']-row['reference_B_confidence']),**result}
   r['C_score_abs_difference']=abs(r['fall_score']-row['reference_C_score']) if result['C_ran'] else None
   if result['C_ran']:assert r['C_score_abs_difference']<1e-5
   records.append(r)
  (OUT/'records.json').write_text(json.dumps(records,indent=2));print(f'Fresh inference {len(records)}/951; actual C calls={len(c_calls)}',flush=True)
 elapsed=time.perf_counter()-start
 hp=[r for r in records if r['cohort']=='HP_FallDetection_val'];har=[r for r in records if r['cohort']=='HAR_activity_probe']
 def count(rr):
  z=Counter()
  for r in rr:
   z['run_C' if r['C_ran'] else 'skip_C']+=1
   if r['label'] in ('Fall','Nonfall'):z[r['label']+('_run' if r['C_ran'] else '_skipped')]+=1
   if r['C_ran']:z['C_Fall' if r['prediction']=='Fall' else 'C_Nonfall']+=1
  return dict(z)
 conditional=Counter()
 for r in hp:
  if r['C_ran']:conditional['tp' if r['label']=='Fall' and r['is_fall'] else 'fn' if r['label']=='Fall' else 'fp' if r['is_fall'] else 'tn']+=1
 assert len(c_calls)==sum(r['C_ran'] for r in records)
 assert set(c_calls)=={r['sample_id'] for r in records if r['C_ran']}
 assert {str(p):sha(p) for p in protected}==before
 summary={'n':len(records),'actual_B_predictions':len(records),'actual_C_model_loads':c_loads,'actual_C_forward_calls':len(c_calls),'C_called_on_skipped_inputs':False,'HP_validation':count(hp),'HP_validation_C_selected_confusion':dict(conditional),'HAR_activity_probe':count(har),'HAR_by_activity':{a:count([r for r in har if r['label']==a]) for a in sorted({r['label'] for r in har})},'max_B_confidence_abs_difference':max(r['B_confidence_abs_difference'] for r in records),'max_C_score_abs_difference':max(r['C_score_abs_difference'] for r in records if r['C_ran']),'weights_threshold_policy_production_replays_unchanged':True,'wall_seconds_including_loads_io_and_inference':elapsed,'timing_note':'Whole diagnostic run duration, not per-sample latency or alarm latency','C_threshold':cfg['threshold'],'B_cutoff':config()['B_confidence_cutoff'],'evidence_limits':['These are already-used validation examples, not independent generalization tests.','Skipped C outputs remain unknown, not Nonfall.','HAR activity labels are not framewise no-fall truth; HAR triggers are not strict false-positive rate.']}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
 (OUT/'integrity.json').write_text(json.dumps(before,indent=2))
 with (OUT/'results.csv').open('w',newline='') as f:
  keys=['cohort','sample_id','label','B_activity','B_confidence','C_ran','fall_score','prediction','state','C_score_abs_difference']
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r.get(k) for k in keys} for r in records)
 print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
