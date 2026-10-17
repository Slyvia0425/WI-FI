from __future__ import annotations

import unittest

from backend.modules.human_activity_recognition.errors import TransportTimeoutError, TransportUnavailableError
from backend.modules.human_activity_recognition.transport import LocalTransport, ResilientTransport


class FakeTransport:
    def __init__(self, error=None, payload=None, fail_times=0):
        self.error = error
        self.payload = payload or {"ok": True}
        self.fail_times = fail_times
        self.calls = 0

    def send(self, payload):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.error or TransportTimeoutError("timeout")
        if self.error and self.calls > self.fail_times:
            raise self.error
        return self.payload


class TransportTest(unittest.TestCase):
    def test_fallback_is_used_only_when_allowed(self):
        primary = FakeTransport(error=TransportTimeoutError("timeout"))
        fallback = FakeTransport(payload={"ok": True, "source": "fallback"})
        result = ResilientTransport(primary, fallback, allow_fallback=True).send({})
        self.assertTrue(result["degraded"])
        self.assertEqual(result["source"], "fallback")
        with self.assertRaises(TransportTimeoutError):
            ResilientTransport(primary, fallback, allow_fallback=False).send({})

    def test_local_transport_uses_handler(self):
        transport = LocalTransport(lambda payload: {"echo": payload["x"]})
        self.assertEqual(transport.send({"x": 3})["echo"], 3)

    def test_both_failures_are_classified(self):
        primary = FakeTransport(error=TransportTimeoutError("timeout"))
        fallback = FakeTransport(error=TransportUnavailableError("down"))
        with self.assertRaises(TransportUnavailableError):
            ResilientTransport(primary, fallback, allow_fallback=True).send({})


class FakeResponse:
    def __init__(self, body=b'{"ok": true}'):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


class RetryOpener:
    def __init__(self, failures=2):
        self.failures = failures
        self.calls = 0

    def __call__(self, request, timeout=0):
        self.calls += 1
        if self.calls <= self.failures:
            raise TimeoutError("timeout")
        return FakeResponse()


class HttpTransportRetryTest(unittest.TestCase):
    def test_http_transport_retries_then_succeeds(self):
        from backend.modules.human_activity_recognition.transport import HttpTransport

        opener = RetryOpener(failures=2)
        transport = HttpTransport("https://example.invalid/predict", retries=2, backoff_seconds=0, opener=opener)
        self.assertTrue(transport.send({})["ok"])
        self.assertEqual(opener.calls, 3)

    def test_http_transport_rejects_unsafe_scheme(self):
        from backend.modules.human_activity_recognition.transport import HttpTransport

        with self.assertRaises(TransportUnavailableError):
            HttpTransport("file:///tmp/data", retries=0).send({})
