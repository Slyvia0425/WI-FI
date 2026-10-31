"""B first. Run frozen C only if same-input B confidence is below0.6."""
import argparse,json
from pathlib import Path
from backend.modules.fall_detection.final_selection_v1.runtime import load_candidate,predict_files,read_amplitude,sha
from backend.modules.fall_detection.activity_reference import segment_identity,activity_reference
from backend.modules.fall_detection.confidence_gate_v1.policy import execute
MODULE=Path(__file__).resolve().parents[1]

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--file',type=Path,required=True)
 group=ap.add_mutually_exclusive_group();group.add_argument('--b-output',type=Path);group.add_argument('--no-b',action='store_true',help='Explicit diagnostic bypass: run C alone, outside the confidence-gated policy')
 ap.add_argument('--output',type=Path);args=ap.parse_args()
 if args.output and args.output.exists():raise FileExistsError('Choose a new output file')
 cfg=json.loads((Path(__file__).parent/'frozen_model.json').read_text());amp=read_amplitude(args.file)
 if amp.shape!=(232,500):raise ValueError('HP raw amplitude[232,500] required')
 identity=segment_identity(amp);base={'file':str(args.file.resolve()),'sample_id':args.file.stem,'recording_id':args.file.stem,'input_id':identity,'device':'HP','threshold':cfg['threshold'],'model_version':cfg['version'],'checkpoint_sha256':cfg['checkpoint_sha256'],'fall_output_index':0}
 b=None;error=None
 if not args.no_b:
  try:
   if args.b_output:b=json.loads(args.b_output.read_text())
   else:
    from backend.modules.human_activity_recognition.integration_v1.runtime import ReleasedActivityModel
    bp=ReleasedActivityModel().predict(amp)
    b={'mock':False,'frames':[{**bp,'input_id':identity,'device':'HP'}]}
  except Exception as exc:error=f'B unavailable ({type(exc).__name__}: {exc})'
 ref=activity_reference(base,b)
 if error:ref['reason']=error
 def infer_c():
  checkpoint=MODULE/cfg['checkpoint']
  if sha(checkpoint)!=cfg['checkpoint_sha256']:raise ValueError('C checkpoint mismatch')
  model,state=load_candidate(checkpoint,'official_resnet18')
  if state['label_mapping']!={'Fall':0,'Nonfall':1}:raise ValueError('C class order mismatch')
  score=predict_files(model,state,'official_resnet18',[{'sample_id':args.file.stem,'file_path':str(args.file),'device':'HP','binary_label':0}])[0]['fall_score']
  positive=score>=cfg['threshold']
  return {'fall_score':score,'prediction':'Fall' if positive else 'Nonfall','is_fall':positive,'state':'suspected' if positive else 'normal'}
 if args.no_b:decision={'C_ran':True,'gate':{'status':'explicit_C_only_diagnostic_bypass'},**infer_c()}
 else:decision=execute(ref,infer_c)
 result={**base,**decision,'activity_reference':ref,'execution_mode':'fresh_inference' if decision['C_ran'] else 'C_not_loaded_or_run','scope':'B confidence routing; skipped C has no binary decision; no training or model threshold change'}
 if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
