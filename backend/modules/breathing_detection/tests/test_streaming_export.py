import unittest
import numpy as np
from backend.modules.breathing_detection.evaluation.export_streaming_dashboard import timeline


class Detector:
    threshold = .05

    def __call__(self, sequence):
        return .9


class StreamingExportTest(unittest.TestCase):
    def source(self):
        t = np.arange(0, 31.1, .021)
        x = np.sin(2 * np.pi * .3 * t[:, None] + np.arange(4)[None] * .2)
        return t, x

    def test_wait_for_actual_final_packet_and_expire(self):
        t, x = self.source()
        result = timeline(t, x, Detector())
        updates = [e for e in result['events'] if 'status' in e]
        self.assertEqual([e['window_end'] for e in updates], [10,15,20,25,30])
        self.assertEqual(updates[-1]['time'], 30.25)
        self.assertIsNotNone(updates[-1]['rate_bpm'])
        self.assertEqual(result['snapshots'][-1]['stream_state'], 'disconnected')
        self.assertIsNone(result['snapshots'][-1]['rate_bpm'])
        self.assertLess(len(result['snapshots']), 15)

    def test_discarded_packets_are_not_reused(self):
        t, x = self.source()
        result = timeline(t, x, Detector(), interrupted=True)
        states = [s['stream_state'] for s in result['snapshots']]
        self.assertIn('disconnected', states)
        self.assertEqual(states.count('warming_up'), 2)
        fresh = [e for e in result['events'] if e.get('generation', 0) > 0 and 'status' in e]
        self.assertTrue(fresh)
        self.assertTrue(all(e['window_start'] >= 12 and e['rate_bpm'] is None for e in fresh))
