import unittest
from backend.modules.fall_detection.confidence_gate_v1.policy import route,execute
class ConfidenceGateTests(unittest.TestCase):
 def ref(self,p):return {'status':'aligned','quality':'good','confidence':p}
 def test_low_confidence_calls_c_once(self):
  calls=[]
  out=execute(self.ref(.5999),lambda:(calls.append('C') or {'fall_score':.9,'prediction':'Fall'}))
  self.assertEqual(calls,['C']);self.assertTrue(out['C_ran']);self.assertEqual(out['prediction'],'Fall')
 def test_boundary_and_high_never_call_c_or_invent_negative(self):
  for confidence in (.6,.9,1.):
   out=execute(self.ref(confidence),lambda:self.fail('C must not be loaded or run'))
   self.assertFalse(out['C_ran']);self.assertIsNone(out['fall_score']);self.assertIsNone(out['prediction']);self.assertIsNone(out['is_fall']);self.assertEqual(out['state'],'not_evaluated')
 def test_missing_unpaired_or_bad_quality_is_unavailable(self):
  for ref in (None,{},dict(self.ref(.2),status='unavailable'),dict(self.ref(.2),quality='degraded')):
   out=execute(ref,lambda:self.fail('No valid B evidence'))
   self.assertEqual(out['gate']['status'],'B_unavailable');self.assertFalse(out['C_ran'])
 def test_invalid_confidence_is_not_a_negative(self):
  for p in (float('nan'),float('inf'),-.1,1.1,True,'0.2'):
   self.assertEqual(route(self.ref(p))['status'],'B_unavailable')
