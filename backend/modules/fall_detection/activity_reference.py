"""Advisory B→C context. No score changes, label weighting, or hard gating.

Replay display time t is never a physical alignment key. Accept only the exact
same canonical amplitude input, or an explicit same-recording/clock window.
"""
from copy import deepcopy
import hashlib
import json
import math

ACTIVITIES={'jumping','running','seated-breathing','walking','wavinghand','transition','unknown'}


def segment_identity(amplitude):
    import numpy as np
    a=np.array(amplitude,dtype='<f4',order='C',copy=True)
    if a.ndim!=2 or not np.isfinite(a).all():raise ValueError('Identity requires finite [features,time] raw amplitudes')
    a[a==0]=0
    body=json.dumps(list(a.shape),separators=(',',':')).encode()+a.tobytes()
    return 'csi-amplitude-f32-v1:'+hashlib.sha256(body).hexdigest()


def _number(value):return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def _match(c,b):
    if c.get('device') and b.get('device') and c['device']!=b['device']:return None
    if c.get('input_id') and b.get('input_id'):
        return 'exact_input' if c['input_id']==b['input_id'] else None
    if not c.get('recording_id') or c.get('recording_id')!=b.get('recording_id'):return None
    if not c.get('clock_id') or c.get('clock_id')!=b.get('clock_id'):return None
    values=[c.get('window_start'),c.get('window_end'),b.get('window_start'),b.get('window_end')]
    if not all(_number(x) for x in values):return None
    cs,ce,bs,be=values
    if cs>=ce or bs>=be or be>ce:return None
    return 'overlapping_recording_window' if min(ce,be)>max(cs,bs) else None


def activity_reference(c,b_payload):
    absent=lambda reason:{'status':'unavailable','reason':reason,'role':'activity_context_only','affects_fall_score':False}
    if not b_payload:return absent('Module B output not supplied')
    if not isinstance(b_payload,dict):return absent('Invalid Module B payload')
    provenance=b_payload.get('provenance') or {}
    if not isinstance(provenance,dict):return absent('Invalid Module B provenance')
    if b_payload.get('mock') is True or provenance.get('trained_artifact_available') is False:
        return absent('Module B fixture is not real paired evidence; trained artifact not supplied')
    candidates=[]
    frames=b_payload.get('frames',[])
    if not isinstance(frames,list):return absent('Invalid Module B frame list')
    for b in frames:
        if not isinstance(b,dict):continue
        match=_match(c,b)
        if match:candidates.append((b,match))
    if not candidates:return absent('No matching input identity or same-clock overlapping recording window')
    exact=[item for item in candidates if item[1]=='exact_input']
    if exact:candidates=exact
    else:
        latest=max(b['window_end'] for b,_ in candidates)
        candidates=[item for item in candidates if item[0]['window_end']==latest]
    for b,_ in candidates:
        if not isinstance(b.get('activity'),str) or b.get('activity') not in ACTIVITIES or not _number(b.get('confidence')) or not 0<=b['confidence']<=1 or not isinstance(b.get('stable'),bool):
            return absent('Invalid Module B activity/confidence/stability')
        if b.get('quality')!='good':return absent('Module B input quality is missing or not good')
    # Exact duplicate contexts are safe; conflicting predictions for one input are ambiguous.
    unique={(b.get('activity'),b.get('confidence'),b.get('stable'),b.get('quality')) for b,_ in candidates}
    if len(unique)>1:return absent('Conflicting Module B outputs for this input; choose a model/run explicitly')
    b,match=candidates[0]
    if b.get('quality')!='good':return absent('Module B input quality is missing or not good')
    if b.get('activity') not in ACTIVITIES or not _number(b.get('confidence')) or not 0<=b['confidence']<=1 or not isinstance(b.get('stable'),bool):
        return absent('Invalid Module B activity/confidence/stability')
    return {'status':'aligned','alignment':match,'activity':b['activity'],'confidence':b['confidence'],
        'stable':b['stable'],'quality':b['quality'],'model_version':b.get('model_version',b_payload.get('model_version','unspecified')),
        'role':'activity_context_only','affects_fall_score':False,
        'decision_note':b.get('decision_note','Activity preference only; no fall class or temporal event confirmed'),
        'note':'Activity context does not confirm or veto the independent ResNet18 fall decision'}


def attach_activity_reference(payload,b_payload):
    out=deepcopy(payload)
    for frame in out.get('frames',[]):frame['activity_reference']=activity_reference(frame,b_payload)
    out['activity_reference_summary']={'aligned':sum(f['activity_reference']['status']=='aligned' for f in out.get('frames',[])),
        'total':len(out.get('frames',[])),'affects_fall_score':False,'display_time_used_for_alignment':False}
    return out
