import copy
import unittest

from backend.modules.breathing_detection.preprocessing.wire_protocol import build_protocol, validate_protocol


class WireProtocolTest(unittest.TestCase):
    def samples(self):
        return [dict(sample_id=f'{subject}-{environment}-{recording}-{window}',
                     recording_id=f'{subject}/{environment}/{recording}', participant_id=subject,
                     environment=environment, room='Room1', posture=f'posture{recording % 2}', assessable=True)
                for subject in ('01', '02', '03') for environment in ('Quiet', 'Interference')
                for recording in range(5) for window in range(2)]

    def test_deterministic_and_subject_recording_disjoint(self):
        samples = self.samples()
        protocol = build_protocol(samples)
        self.assertEqual(protocol, build_protocol(samples))
        validate_protocol(protocol, samples)
        for fold in protocol['folds'].values():
            self.assertTrue(set(fold['train']).isdisjoint(fold['validation']))
            self.assertEqual({samples[i]['participant_id'] for i in fold['assessment']}, {fold['held_out_subject']})

    def test_rejects_recording_leakage_and_changed_metadata(self):
        samples = self.samples()
        protocol = build_protocol(samples)
        bad = copy.deepcopy(protocol)
        fold = bad['folds']['subject_01']
        index = fold['validation'].pop()
        fold['train'].append(index)
        with self.assertRaisesRegex(ValueError, 'Recording leakage'):
            validate_protocol(bad, samples)
        samples[0]['assessable'] = False
        with self.assertRaisesRegex(ValueError, 'metadata'):
            validate_protocol(protocol, samples)
