"""Pure routing contract. A C callback is never called on skipped inputs."""
import json,math
from pathlib import Path

def config():return json.loads(Path(__file__).with_name('policy.json').read_text())

def route(reference):
 cutoff=config()['B_confidence_cutoff']
 base={'policy':'B_low_confidence_C_v1','B_confidence_cutoff':cutoff,'comparison':'confidence < cutoff'}
 if not isinstance(reference,dict) or reference.get('status')!='aligned' or reference.get('quality')!='good':
  return {**base,'run_C':False,'status':'B_unavailable','reason':'Valid same-input B evidence is required'}
 p=reference.get('confidence')
 if isinstance(p,bool) or not isinstance(p,(int,float)) or not math.isfinite(p) or not 0<=p<=1:
  return {**base,'run_C':False,'status':'B_unavailable','reason':'Invalid B confidence'}
 run=p<cutoff
 return {**base,'B_confidence':p,'run_C':run,'status':'B_low_confidence' if run else 'B_high_confidence','reason':'B confidence <0.6: evaluate C' if run else 'B confidence >=0.6: C not evaluated'}

def execute(reference,c_predict):
 decision=route(reference)
 if decision['run_C']:
  return {'gate':decision,'C_ran':True,**c_predict()}
 return {'gate':decision,'C_ran':False,'fall_score':None,'prediction':None,'is_fall':None,'state':'not_evaluated'}
