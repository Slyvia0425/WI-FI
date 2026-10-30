"""Bounded timestamp-aware streaming inference with an independent watchdog."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from typing import Callable

import numpy as np

from ..preprocessing.unified_rate import estimate_unified_rate, ESTIMATOR_ID
from ..preprocessing.temporal import extract_interpolated_subcarriers
from .temporal_risk import TemporalInstabilityTracker


@dataclass(frozen=True)
class StreamConfig:
    detection_seconds: float = 10.
    rate_seconds: float = 30.
    stride_seconds: float = 5.
    target_rate_hz: float = 30.
    max_gap_seconds: float = .1
    disconnect_seconds: float = 2.
    breathing_threshold: float = .054542455822229385

    def __post_init__(self):
        values = (self.detection_seconds, self.rate_seconds, self.stride_seconds,
                  self.target_rate_hz, self.max_gap_seconds, self.disconnect_seconds)
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError('Timing parameters must be finite and positive')
        if self.rate_seconds < self.detection_seconds or self.disconnect_seconds <= self.max_gap_seconds:
            raise ValueError('Invalid rate-window or disconnect duration')
        if self.rate_seconds != 30. or self.target_rate_hz != 30.:
            raise ValueError('Unified rate estimator requires 30-second windows at 30 Hz')
        if not 0 < self.breathing_threshold < 1:
            raise ValueError('Threshold must lie in (0,1)')
        for duration in (self.detection_seconds, self.rate_seconds):
            if not np.isclose(duration * self.target_rate_hz, round(duration * self.target_rate_hz)):
                raise ValueError('Window size must contain an integer number of points')


class BreathingStream:
    """Accept raw ``[time,subcarrier]`` amplitudes in strictly ordered batches.

    The detector receives standardized/interpolated ``[300,232]`` inputs.
    Arrival times must use a single monotonic clock, independent of sensor time.
    Call ``poll`` regularly even when packets stop arriving.
    """

    def __init__(self, detector: Callable[[np.ndarray], float], config: StreamConfig | None = None):
        self.config = config or StreamConfig()
        self.detector = detector
        self.tracker = TemporalInstabilityTracker(self.config.breathing_threshold)
        self.generation = 0
        self.last_result = None
        self.last_arrival = None
        self.clock_seen = None
        self.state = 'waiting'
        self.reason = 'insufficient_data'
        self._clear()

    def _clear(self):
        self.points = deque()
        self.anchor = None
        self.next_end = None
        self.features = None
        self.tracker.reset()

    def reset(self, reason='manual_reset'):
        self._clear()
        self.generation += 1
        self.last_arrival = None
        self.state = 'warming_up'
        self.reason = reason

    def _clock(self, now):
        now = time.monotonic() if now is None else float(now)
        if not np.isfinite(now) or (self.clock_seen is not None and now < self.clock_seen):
            raise ValueError('Arrival/watchdog clock must be finite and monotonic')
        self.clock_seen = now
        return now

    def poll(self, now=None):
        now = self._clock(now)
        if self.last_arrival is not None and now - self.last_arrival >= self.config.disconnect_seconds and self.state != 'disconnected':
            self.reset('arrival_timeout')
            self.state = 'disconnected'
        active = self.last_result is not None and self.last_result['generation'] == self.generation
        return {'stream_state': self.state, 'reason': self.reason, 'generation': self.generation,
                'status': self.last_result['status'] if active else 'not_assessable',
                'rate_bpm': self.last_result['rate_bpm'] if active else None,
                'last_result': self.last_result, 'last_result_stale': self.last_result is not None and not active}

    def _window(self, end, duration):
        times = np.asarray([p[0] for p in self.points])
        start = end - duration
        if times[0] > start + 1e-7 or times[-1] < end - 1e-7:
            return None, 'insufficient_data'
        count = round(duration * self.config.target_rate_hz)
        target = start + np.arange(count) / self.config.target_rate_hz
        relevant = (times[:-1] <= target[-1]) & (times[1:] >= target[0])
        if np.any(np.diff(times)[relevant] > self.config.max_gap_seconds):
            return None, 'timestamp_gap'
        values = np.stack([p[1] for p in self.points])
        x = np.asarray([np.interp(target, times, values[:, k]) for k in range(values.shape[1])], dtype=np.float32)[None]
        return x, None

    def _infer(self, end):
        c = self.config
        x, quality = self._window(end, c.detection_seconds)
        probability = instability = rate = rate_details = None
        if quality:
            status = 'not_assessable'
            self.tracker.reset()
        else:
            try:
                sequence = extract_interpolated_subcarriers(x)
                probability = float(self.detector(sequence))
                if not np.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError('Invalid detector score')
                instability = self.tracker.update(probability)
                status = 'breathing' if probability >= c.breathing_threshold else 'none'
            except ValueError:
                status, quality, probability = 'not_assessable', 'invalid_signal_or_score', None
                self.tracker.reset()
        rate_x, rate_reason = self._window(end, c.rate_seconds)
        if not rate_reason:
            if status != 'breathing':
                rate_reason = 'insufficient_breathing_evidence'
            else:
                try:
                    rate_details = estimate_unified_rate(rate_x, c.target_rate_hz)
                    rate = rate_details['bpm']
                    rate_reason = 'experimental_unified'
                except ValueError:
                    rate_reason = 'invalid_rate_signal'
        result = {'generation': self.generation, 'window_start': end - c.detection_seconds,
                  'window_end': end, 'status': status, 'reason': quality or 'model_evidence',
                  'breathing_probability': probability, 'temporal_instability': instability,
                  'temporal_gate_enabled': False, 'rate_bpm': rate, 'rate_reason': rate_reason,
                  'rate_estimator_id': ESTIMATOR_ID, 'rate_details': rate_details,
                  'rate_window_seconds': c.rate_seconds}
        self.last_result = result
        self.state, self.reason = 'active', quality or 'model_evidence'
        return result

    def push(self, timestamps, amplitudes, *, arrival_time=None):
        now = self._clock(arrival_time)
        t = np.asarray(timestamps, dtype=float)
        x = np.asarray(amplitudes, dtype=np.float32)
        if (t.ndim != 1 or x.ndim != 2 or len(t) != len(x) or not len(t)
                or x.shape[1] < 2 or not np.isfinite(t).all() or not np.isfinite(x).all()
                or not np.all(np.diff(t) > 0)):
            self.reset('invalid_batch')
            raise ValueError('Require finite, strictly increasing timestamps and aligned [time,K>=2] amplitudes')
        transitions = []
        if self.last_arrival is not None and now - self.last_arrival >= self.config.disconnect_seconds:
            self.reset('arrival_timeout')
            transitions.append({'event': 'reset', 'reason': 'arrival_timeout', 'generation': self.generation})
        if self.points and (t[0] <= self.points[-1][0] or x.shape[1] != self.features):
            reason = 'sensor_time_reset' if t[0] <= self.points[-1][0] else 'subcarrier_structure_changed'
            self.reset(reason)
            transitions.append({'event': 'reset', 'reason': reason, 'generation': self.generation})
        for sensor_time, values in zip(t, x):
            if self.points and sensor_time - self.points[-1][0] >= self.config.disconnect_seconds:
                self.reset('sensor_time_gap')
                transitions.append({'event': 'reset', 'reason': 'sensor_time_gap', 'generation': self.generation})
            if self.anchor is None:
                self.anchor = float(sensor_time)
                self.next_end = self.anchor + self.config.detection_seconds
                self.features = x.shape[1]
                self.state = 'warming_up'
            self.points.append((float(sensor_time), values.copy()))
            while len(self.points) > 2 and self.points[1][0] < sensor_time - self.config.rate_seconds - self.config.max_gap_seconds:
                self.points.popleft()
            while sensor_time >= self.next_end - 1e-7:
                transitions.append(self._infer(self.next_end))
                self.next_end += self.config.stride_seconds
        self.last_arrival = now
        return transitions
