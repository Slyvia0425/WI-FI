"""Controlled interface checks, not model-performance experiments."""
import unittest
from .gate import run_cascade,ModelNotReadyError

class GateChecks(unittest.TestCase):
 def setUp(self):self.calls=[]
 def invoke(self,activity='Fall',labels=('running','Fall'),quality='good',b_id='input-A',c_id='input-A',c_decision='Fall'):
  def b():
   self.calls.append('B');return {'activity':activity,'input_id':b_id,'quality':quality,'confidence':.4,'model_version':'test-double-B'}
  def c():
   self.calls.append('C');return {'input_id':c_id,'prediction':c_decision,'fall_score':.8 if c_decision=='Fall' else .1,'threshold':.5,'model_version':'test-double-C'}
  return run_cascade(input_id='input-A',b_labels=labels,b_predict=b,c_predict=c)
 def test_original_five_class_model_refused_before_any_inference(self):
  with self.assertRaises(ModelNotReadyError):self.invoke(labels=('running','walking','jumping','seated-breathing','wavinghand'))
  self.assertEqual(self.calls,[])
 def test_ordinary_activity_never_runs_c_or_invents_a_c_negative(self):
  out=self.invoke(activity='running');self.assertEqual(self.calls,['B']);self.assertFalse(out['C_ran']);self.assertIsNone(out['C_prediction']);self.assertIsNone(out['C_fall_score'])
 def test_fall_runs_c_without_an_invented_new_confidence_cutoff(self):
  out=self.invoke();self.assertEqual(self.calls,['B','C']);self.assertTrue(out['C_ran']);self.assertEqual(out['C_prediction'],'Fall')
 def test_c_may_reject_b_fall_candidate(self):
  out=self.invoke(c_decision='Nonfall');self.assertEqual(out['C_prediction'],'Nonfall');self.assertTrue(out['C_ran'])
 def test_unknown_b_is_unavailable_not_negative(self):
  out=self.invoke(activity='unknown');self.assertEqual(out['status'],'B_evidence_unavailable');self.assertIsNone(out['B_fall_candidate']);self.assertEqual(self.calls,['B'])
 def test_bad_quality_does_not_run_c(self):
  out=self.invoke(quality='invalid');self.assertEqual(self.calls,['B']);self.assertEqual(out['status'],'B_evidence_unavailable')
 def test_b_identity_mismatch_never_runs_c(self):
  with self.assertRaises(ValueError):self.invoke(b_id='other')
  self.assertEqual(self.calls,['B'])
 def test_c_identity_mismatch_is_rejected(self):
  with self.assertRaises(ValueError):self.invoke(c_id='other')

if __name__=='__main__':unittest.main()
