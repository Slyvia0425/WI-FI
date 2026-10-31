"""B screens for Fall, then C confirms on the same input.

This optional adapter is not wired into the existing replay or CLI. A real B
model with a trained Fall class is required; the shipped five-class B is refused.
Callbacks must infer from the same raw input and return its canonical input_id.
"""
import math

class ModelNotReadyError(ValueError):
    pass


def run_cascade(*, input_id, b_labels, b_predict, c_predict, fall_label='Fall'):
    """Lazy call C only when an eligible B model actually returns its Fall class.

    b_predict() -> {input_id, activity, quality, model_version, confidence}
    c_predict() -> {input_id, prediction, fall_score, threshold, model_version}
    No additional confidence threshold is introduced by this gate.
    """
    if fall_label not in b_labels:
        raise ModelNotReadyError('B model does not contain a trained '+fall_label+' class; cascade is not ready')
    if not isinstance(input_id,str) or not input_id:
        raise ValueError('Canonical input identity is required')
    b=b_predict()
    if not isinstance(b,dict) or b.get('input_id')!=input_id:
        raise ValueError('B input identity mismatch')
    if b.get('mock') is True:
        raise ValueError('A fixture cannot trigger the real cascade')
    activity=b.get('activity')
    base={'input_id':input_id,'B_output':b,'C_ran':False,'C_output':None,
          'C_prediction':None,'C_fall_score':None}
    # Missing/invalid/unknown evidence is unavailable, not a negative decision.
    if b.get('quality')!='good' or activity in ('unknown','transition',None):
        return {**base,'status':'B_evidence_unavailable','B_fall_candidate':None}
    if activity not in b_labels:
        raise ValueError('B output is outside its declared trained classes')
    confidence=b.get('confidence')
    if isinstance(confidence,bool) or not isinstance(confidence,(int,float)) or not math.isfinite(confidence) or not 0<=confidence<=1:
        raise ValueError('Invalid B confidence')
    if activity!=fall_label:
        return {**base,'status':'not_triggered_by_B','B_fall_candidate':False}
    c=c_predict()
    if not isinstance(c,dict) or c.get('input_id')!=input_id:
        raise ValueError('C input identity mismatch')
    if c.get('prediction') not in ('Fall','Nonfall'):
        raise ValueError('C must provide its actual binary decision')
    return {**base,'status':'C_evaluated','B_fall_candidate':True,'C_ran':True,
            'C_output':c,'C_prediction':c['prediction'],'C_fall_score':c.get('fall_score')}
