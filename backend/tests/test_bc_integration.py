"""Verify imported B and real C replay remain distinct and faithfully exported."""
import csv
import json
from pathlib import Path
import unittest

from backend.api.routes import dispatch
from backend.modules.fall_detection.export_demo import build_replay
from backend.modules.human_activity_recognition.contract import validate_frame_values
from backend.shared.contract import validate_replay

ROOT=Path(__file__).resolve().parents[2]
FALL_REPORT=ROOT/'backend/modules/fall_detection/reports/tcs_cross_subject_v1/comparison_seed42_v1/evaluation'


class BCIntegrationTest(unittest.TestCase):
    def test_both_routes_preserve_their_actual_data_status(self):
        sb,b=dispatch('GET','/api/human_activity_recognition/replay')
        sc,c=dispatch('GET','/api/fall_detection/replay')
        self.assertEqual((sb,sc),(200,200))
        self.assertIs(b['mock'],False)
        self.assertIs(c['mock'],False)
        self.assertTrue(b['provenance']['trained_artifact_available'])
        self.assertTrue(b['provenance']['inference_reproduced_locally'])
        self.assertEqual(c['split'],'validation')
        self.assertFalse(c['provenance']['joint_ground_truth'])

    def test_real_b_is_paired_by_input_identity_without_changing_c(self):
        _,b=dispatch('GET','/api/human_activity_recognition/replay')
        _,c=dispatch('GET','/api/fall_detection/replay')
        previous=json.loads((ROOT/'backend/modules/fall_detection/outputs/before_real_B_v1.json').read_text())
        self.assertEqual(len(b['frames']),926)
        self.assertEqual(c['activity_reference_summary']['aligned'],926)
        for bf,cf,old in zip(b['frames'],c['frames'],previous['frames'],strict=True):
            self.assertEqual(bf['input_id'],cf['input_id'])
            self.assertEqual(cf['activity_reference']['alignment'],'exact_input')
            self.assertEqual(cf['activity_reference']['activity'],bf['activity'])
            for key in ('sample_id','input_id','threshold'):
                self.assertEqual(cf[key],old[key])
            if cf['C_ran']:
                self.assertEqual(cf['fall_score'],old['fall_score'])
            else:
                self.assertIsNone(cf['fall_score']);self.assertIsNone(cf['prediction']);self.assertEqual(cf['state'],'not_evaluated')

    def test_no_five_class_accuracy_invented_on_fall_detection(self):
        _,b=dispatch('GET','/api/human_activity_recognition/replay')
        self.assertNotIn('confusion',b['metrics'])
        self.assertTrue(all(f['activity_ground_truth_available'] is False for f in b['frames']))
        self.assertEqual({f['activity'] for f in b['frames']} - {'jumping','running','seated-breathing','walking','wavinghand'},set())

    def test_b_values_and_export_contract(self):
        _,b=dispatch('GET','/api/human_activity_recognition/replay')
        self.assertEqual(validate_replay('human_activity_recognition',b),[])
        self.assertEqual(validate_frame_values(b),[])

    def test_all_twelve_c_recordings_match_original_predictions(self):
        if not FALL_REPORT.exists():self.skipTest('Original research scores not included; route tests still run')
        payload=build_replay()
        with (FALL_REPORT/'SA_CT_standardized_GCV_windows.csv').open() as f:rows=list(csv.DictReader(f))
        expected={(r['member'],float(r['window_end_seconds'])):float(r['fall_score']) for r in rows}
        self.assertEqual(len(payload['recordings']),12)
        self.assertEqual(sum(len(r['frames']) for r in payload['recordings']),len(expected))
        for record in payload['recordings']:
            self.assertEqual(validate_replay('fall_detection',{'frames':record['frames']}),[])
            last=False
            for frame in record['frames']:
                score=expected[(record['id'],frame['t'])]
                self.assertEqual(frame['fall_score'],score)
                self.assertEqual(frame['threshold'],0.5)
                positive=score>=0.5
                self.assertEqual(frame['state'],'suspected' if positive else 'normal')
                self.assertEqual(frame['alarm_onset'],positive and not last)
                last=positive
        self.assertEqual(sum(r['background_alarm_episodes'] for r in payload['recordings']),2)

    def test_api_and_offline_bundle_are_identical(self):
        for name in ('human_activity_recognition','fall_detection'):
            status,payload=dispatch('GET',f'/api/{name}/replay')
            bundled=json.loads((ROOT/f'frontend/src/modules/{name}/data/replay.json').read_text())
            self.assertEqual(status,200);self.assertEqual(payload,bundled)

    def test_routing_coverage_and_skipped_falls_are_explicit(self):
        _,payload=dispatch('GET','/api/fall_detection/replay')
        self.assertEqual(payload['gate_summary']['run_C'],621)
        self.assertEqual(payload['gate_summary']['skip_C'],305)
        self.assertEqual(payload['gate_summary']['Fall_skipped'],51)

    def test_resnet_primary_metrics_preserve_original_hp_validation(self):
        _,payload=dispatch('GET','/api/fall_detection/replay')
        confusion=payload['original_C_metrics']['confusion']
        self.assertEqual(confusion['labels'],['Nonfall','Fall'])
        self.assertEqual(confusion['matrix'],[[513,23],[6,384]])
        self.assertAlmostEqual(payload['threshold'],0.574872225522995)
        self.assertEqual(payload['model_name'],'CSI-Bench official ResNet18')

if __name__=='__main__':unittest.main()
