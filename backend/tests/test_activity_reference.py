"""B context must align to real input identity and never veto or amplify C."""
from copy import deepcopy
import unittest
from backend.modules.fall_detection.activity_reference import activity_reference,attach_activity_reference

class ActivityReferenceTest(unittest.TestCase):
    def setUp(self):
        self.c={'input_id':'csi-amplitude-f32-v1:'+'0'*64,'device':'HP','t':5,'fall_score':.9,'threshold':.57,'state':'suspected'}
        self.b={'input_id':self.c['input_id'],'device':'HP','t':5,'activity':'walking','confidence':.9,'stable':True,'quality':'good'}
    def wrap(self,b=None,**kw):return {'frames':[self.b if b is None else b],'mock':False,**kw}
    def test_matched_context_cannot_change_c_score_or_state(self):
        original={'frames':[self.c]};before=deepcopy(original)
        out=attach_activity_reference(original,self.wrap())
        self.assertEqual(original,before)
        self.assertEqual(out['frames'][0]['fall_score'],.9)
        self.assertEqual(out['frames'][0]['state'],'suspected')
        self.assertEqual(out['frames'][0]['activity_reference']['activity'],'walking')
        self.assertFalse(out['activity_reference_summary']['affects_fall_score'])
    def test_low_c_score_not_promoted_by_activity_transition(self):
        c={**self.c,'fall_score':.1,'state':'normal'}
        b={**self.b,'activity':'transition','stable':False}
        out=attach_activity_reference({'frames':[c]},self.wrap(b))
        self.assertEqual(out['frames'][0]['state'],'normal')
        self.assertEqual(out['frames'][0]['fall_score'],.1)
    def test_missing_b_is_explicit(self):
        self.assertEqual(activity_reference(self.c,None)['status'],'unavailable')
    def test_matching_display_time_is_not_alignment(self):
        b={**self.b,'input_id':'different'}
        self.assertEqual(activity_reference(self.c,self.wrap(b))['status'],'unavailable')
    def test_mock_cannot_become_evidence_even_with_matching_id(self):
        self.assertEqual(activity_reference(self.c,self.wrap(mock=True))['status'],'unavailable')
    def test_bad_quality_or_device_is_rejected(self):
        for key,value in [('quality','degraded'),('quality',None),('device','ESP32')]:
            self.assertEqual(activity_reference(self.c,self.wrap({**self.b,key:value}))['status'],'unavailable')
    def test_invalid_payload_never_breaks_c(self):
        cases=[[],{'frames':{}},{'frames':[None]},{'frames':[dict(self.b,confidence={'bad':1})]},
               {'frames':[dict(self.b,confidence=float('nan'))]},{'frames':[dict(self.b,activity={})]}]
        for b in cases:
            out=attach_activity_reference({'frames':[self.c]},b)
            self.assertEqual(out['frames'][0]['fall_score'],.9)
            self.assertEqual(out['frames'][0]['activity_reference']['status'],'unavailable')
    def test_conflicting_outputs_require_explicit_choice(self):
        payload={'mock':False,'frames':[self.b,{**self.b,'activity':'running'}]}
        self.assertEqual(activity_reference(self.c,payload)['status'],'unavailable')
    def test_same_clock_overlapping_past_window_is_accepted(self):
        c={'recording_id':'R','clock_id':'nic','window_start':10.,'window_end':15.,'device':'HP'}
        b={k:v for k,v in self.b.items() if k!='input_id'}
        b.update(recording_id='R',clock_id='nic',window_start=9.,window_end=14.)
        self.assertEqual(activity_reference(c,self.wrap(b))['alignment'],'overlapping_recording_window')
    def test_future_stale_or_different_clock_context_is_rejected(self):
        c={'recording_id':'R','clock_id':'nic','window_start':10.,'window_end':15.}
        base={k:v for k,v in self.b.items() if k!='input_id'}
        base.update(recording_id='R',clock_id='nic',window_start=9.,window_end=14.)
        for extra in [{'window_start':12.,'window_end':16.},{'window_start':0.,'window_end':5.},{'clock_id':'another'}]:
            self.assertEqual(activity_reference(c,self.wrap({**base,**extra}))['status'],'unavailable')
    def test_unreadable_b_file_does_not_break_c_api(self):
        from unittest.mock import patch
        from backend.modules.fall_detection import api
        payload={'mode':'independent_segment_validation','frames':[self.c]}
        with patch.object(api,'read_replay',side_effect=[payload,ValueError('invalid B JSON')]):
            result=api.replay()
        self.assertEqual(result['frames'][0]['fall_score'],.9)
        self.assertEqual(result['frames'][0]['activity_reference']['status'],'unavailable')

    def test_latest_overlapping_window_is_selected_without_future(self):
        c={'recording_id':'R','clock_id':'nic','window_start':10.,'window_end':15.}
        base={k:v for k,v in self.b.items() if k!='input_id'}
        base.update(recording_id='R',clock_id='nic',window_start=9.,window_end=12.,activity='running')
        newer={**base,'window_start':11.,'window_end':14.,'activity':'walking'}
        r=activity_reference(c,{'mock':False,'frames':[base,newer]})
        self.assertEqual(r['status'],'aligned');self.assertEqual(r['activity'],'walking')

if __name__=='__main__':unittest.main()
