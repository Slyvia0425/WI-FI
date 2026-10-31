import unittest

import numpy as np

from backend.modules.breathing_detection.inference.streaming import BreathingStream


def signal(start=0., end=40., features=4):
    t = np.arange(round((end - start) * 50) + 1) / 50 + start
    x = np.sin(2 * np.pi * .3 * t[:, None] + np.arange(features)[None] * .2).astype(np.float32)
    return t, x


class StreamingTest(unittest.TestCase):
    def stream(self):
        def detector(x):
            self.assertEqual(x.shape, (300, 232))
            return .9
        return BreathingStream(detector)

    def test_cadence_warmup_rate_and_batch_invariance(self):
        t, x = signal()
        a, b = self.stream(), self.stream()
        entire = a.push(t, x, arrival_time=0)
        chunks = []
        for start in range(0, len(t), 13):
            chunks.extend(b.push(t[start:start + 13], x[start:start + 13], arrival_time=t[start]))
        self.assertEqual(entire, chunks)
        self.assertEqual([r['window_end'] for r in entire], [10, 15, 20, 25, 30, 35, 40])
        self.assertTrue(all(r['rate_bpm'] is None for r in entire[:4]))
        self.assertAlmostEqual(entire[4]['rate_bpm'], 18, delta=1)
        self.assertIsNone(entire[1]['temporal_instability'])
        self.assertIsNotNone(entire[2]['temporal_instability'])
        self.assertFalse(entire[-1]['temporal_gate_enabled'])

    def test_short_gap_rejects_only_affected_windows_and_history_resets(self):
        t, x = signal()
        keep = ~((t > 11.) & (t < 11.5))
        result = self.stream().push(t[keep], x[keep], arrival_time=0)
        self.assertEqual(result[0]['status'], 'breathing')
        self.assertEqual(result[1]['reason'], 'timestamp_gap')
        self.assertEqual(result[2]['status'], 'not_assessable')
        self.assertEqual(result[3]['status'], 'breathing')
        self.assertIsNone(result[3]['temporal_instability'])
        self.assertEqual(result[4]['rate_reason'], 'timestamp_gap')

    def test_watchdog_expires_output_and_recovery_warms_up(self):
        t, x = signal(end=30)
        stream = self.stream()
        stream.push(t, x, arrival_time=0)
        self.assertEqual(stream.poll(1)['status'], 'breathing')
        stale = stream.poll(2)
        self.assertEqual(stale['stream_state'], 'disconnected')
        self.assertTrue(stale['last_result_stale'])
        self.assertIsNone(stale['rate_bpm'])
        generation = stream.generation
        stream.poll(3)
        self.assertEqual(stream.generation, generation)
        t, x = signal(start=31, end=40)
        self.assertEqual(stream.push(t, x, arrival_time=3), [])
        self.assertEqual(stream.poll(3)['stream_state'], 'warming_up')
        self.assertEqual(stream.poll(3)['status'], 'not_assessable')
        t, x = signal(start=40.02, end=61)
        recovered = stream.push(t, x, arrival_time=3.5)
        self.assertEqual(recovered[0]['window_end'], 41)
        self.assertTrue(all(e['rate_bpm'] is None for e in recovered[:-1]))
        self.assertIsNotNone(recovered[-1]['rate_bpm'])
        self.assertFalse(stream.poll(3.5)['last_result_stale'])

    def test_long_sensor_gap_never_bridges_old_and_new_data(self):
        stream = self.stream()
        t, x = signal(end=10)
        stream.push(t, x, arrival_time=0)
        t, x = signal(start=13, end=22)
        events = stream.push(t, x, arrival_time=1)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['reason'], 'sensor_time_gap')
        self.assertTrue(stream.poll(1)['last_result_stale'])
        self.assertIsNone(stream.poll(1)['rate_bpm'])

    def test_regression_structure_change_invalid_values_and_clock(self):
        stream = self.stream()
        t, x = signal(end=10)
        stream.push(t, x, arrival_time=0)
        events = stream.push(t[:5], x[:5], arrival_time=1)
        self.assertEqual(events[0]['reason'], 'sensor_time_reset')
        stream.push(t[5:10], np.tile(x[5:10], (1, 2)), arrival_time=1)
        self.assertEqual(stream.reason, 'subcarrier_structure_changed')
        with self.assertRaises(ValueError):
            stream.push([1, 1], np.ones((2, 4)), arrival_time=1)
        self.assertEqual(len(stream.points), 0)
        with self.assertRaises(ValueError):
            stream.poll(.5)

    def test_nonfinite_data_clear_current_results(self):
        stream = self.stream()
        t, x = signal(end=10)
        stream.push(t, x, arrival_time=0)
        invalid = np.ones((2, 4), dtype=np.float32)
        invalid[0, 0] = np.nan
        with self.assertRaises(ValueError):
            stream.push([10.02, 10.04], invalid, arrival_time=1)
        snapshot = stream.poll(1)
        self.assertTrue(snapshot['last_result_stale'])
        self.assertEqual(snapshot['status'], 'not_assessable')
        self.assertIsNone(snapshot['rate_bpm'])

    def test_unstable_predictions_are_diagnostic_not_rejected(self):
        probabilities = iter([.9, .001, .9])
        stream = BreathingStream(lambda x: next(probabilities))
        t, x = signal(end=20)
        result = stream.push(t, x, arrival_time=0)
        self.assertEqual(result[-1]['temporal_instability'], 1.)
        self.assertEqual(result[-1]['status'], 'breathing')

    def test_receive_timeout_without_poll_also_resets(self):
        stream = self.stream()
        t, x = signal(end=10)
        stream.push(t, x, arrival_time=0)
        events = stream.push(t[-1:] + .02, x[-1:], arrival_time=3)
        self.assertEqual(events[0]['reason'], 'arrival_timeout')
        self.assertEqual(stream.poll(3)['status'], 'not_assessable')

    def test_buffer_is_time_bounded(self):
        stream = self.stream()
        t, x = signal(end=100)
        stream.push(t, x, arrival_time=0)
        self.assertLess(len(stream.points), 1600)
