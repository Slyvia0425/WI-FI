import unittest
from backend.modules.breathing_detection.evaluation.analyze_motion_pollution import matched_controls


def row(sample_id,record='r1',person='01',environment='Interference',prediction=12):
    return dict(sample_id=sample_id,recording_id=record,participant_id=person,posture='Sitting',room='Room1',
                environment=environment,reference_bpm=12,predictions={'unified':prediction})


class MotionMatchingTest(unittest.TestCase):
    def test_same_record_priority_and_id_order_not_best_error(self):
        target=row('target',prediction=30)
        controls=[row('z'),row('a',prediction=13),row('0',record='other'),row('bad',prediction=28)]
        tier,selected=matched_controls(target,controls)
        self.assertEqual(tier,'same_recording')
        self.assertEqual([r['sample_id'] for r in selected],['a','z'])

    def test_same_environment_preferred_then_explicit_environment_fallback(self):
        target=row('target',prediction=30)
        quiet=row('a',record='quiet',environment='Quiet')
        same=row('b',record='other')
        tier,selected=matched_controls(target,[quiet,same])
        self.assertEqual(tier,'same_person_posture_room_environment')
        self.assertEqual([r['sample_id'] for r in selected],['b'])
        tier,selected=matched_controls(target,[quiet])
        self.assertEqual(tier,'same_person_posture_room')
        self.assertEqual(selected[0]['environment'],'Quiet')

    def test_missing_person_match_not_filled_with_unrelated_control(self):
        tier,selected=matched_controls(row('target',prediction=30),[row('other',record='r2',person='02')])
        self.assertEqual(tier,'unmatched')
        self.assertEqual(selected,[])
