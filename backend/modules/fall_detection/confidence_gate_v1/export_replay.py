"""Apply routing to preserved validation scores. This is a replay simulation, not621 new inferences."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from backend.shared.export import read_replay,write_replay
from backend.modules.fall_detection.activity_reference import attach_activity_reference
from .policy import route,config
M=Path(__file__).resolve().parents[1]

def main():
 archive=M/'outputs/before_confidence_gate_v1.json'
 if not archive.exists():archive.write_text(json.dumps(read_replay('fall_detection'),indent=1))
 p=attach_activity_reference(json.loads(archive.read_text()),read_replay('human_activity_recognition'))
 p['original_C_metrics']=deepcopy(p['metrics']);p['original_C_metrics_values']=deepcopy(p['metrics_values'])
 counts=Counter();conditional=Counter()
 for f in p['frames']:
  g=route(f.get('activity_reference'));f['gate']=g;f['C_ran']=g['run_C']
  f['execution_mode']='cached_validation_score_selected' if g['run_C'] else 'not_run_by_gate'
  counts['run_C' if g['run_C'] else 'skip_C']+=1
  counts[f['ground_truth']+('_run' if g['run_C'] else '_skipped')]+=1
  if g['run_C']:
   f['is_fall']=f['prediction']=='Fall'
   conditional['tp' if f['ground_truth']=='Fall' and f['is_fall'] else 'fn' if f['ground_truth']=='Fall' else 'fp' if f['is_fall'] else 'tn']+=1
  else:
   f.update(fall_score=None,prediction=None,is_fall=None,state='not_evaluated',correct=None)
 p['mode']='b_low_confidence_gate_validation';p['gate_policy']=config();p['gate_summary']={**dict(counts),'conditional_C_confusion':dict(conditional)}
 p['provenance']['routing']='Post-hoc validation replay simulation using already saved B/C scores; not new inference for all926 or an independent test'
 p['metrics_values']=dict(conditional)
 p['metrics']={'table':[{'scope':'Post-hoc validation routing replay','measure':'B confidence cutoff','value':'C only if B confidence <0.6'},
 {'scope':'926 HP validation segments','measure':'C selected / not evaluated','value':f"{counts['run_C']} / {counts['skip_C']}"},
 {'scope':'390 actual Fall segments','measure':'Fall segments skipped by B gate','value':counts['Fall_skipped']},
 {'scope':'C-selected subset only','measure':'TP / FP / FN / TN','value':f"{conditional['tp']} / {conditional['fp']} / {conditional['fn']} / {conditional['tn']}"},
 {'scope':'Frozen C','measure':'C threshold','value':p['threshold']},
 {'scope':'Evidence limitation','measure':'Skipped C outputs','value':'Unknown / not evaluated; not Nonfall'},
 {'scope':'Evaluation scope','measure':'Status','value':'Exploratory validation replay; not a new held-out benchmark'}],
 'confusion':{'labels':['Nonfall','Fall'],'matrix':[[conditional['tn'],conditional['fp']],[conditional['fn'],conditional['tp']]]}}
 p['experiments']['table'].insert(0,{'pipeline':'B confidence <0.6 → C','role':'Active routing; cached validation replay'})
 p['preprocessing']['tiles'][-1]={'label':'C routing','value':'B confidence <0.6 only'}
 write_replay('fall_detection',p,bundle=True)
 report=M/'reports/confidence_gate_v1';report.mkdir(exist_ok=True)
 (report/'validation_routing.json').write_text(json.dumps({'policy':config(),'counts':dict(counts),'C_selected_subset':dict(conditional),'score_source':'Previously saved frozen B/C validation predictions; no new full-set inference','skipped_C_is_not_Nonfall':True},indent=2))
 print(json.dumps(p['gate_summary'],indent=2))
if __name__=='__main__':main()
