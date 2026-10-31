from __future__ import annotations

import unittest

from backend.modules.human_activity_recognition.errors import InputValidationError
from backend.modules.human_activity_recognition.resource import ResourceGuard, ResourceLimits


class ResourceLimitsTest(unittest.TestCase):
    def test_batch_and_input_limits(self):
        guard = ResourceGuard(
            ResourceLimits(
                max_batch_size=2,
                max_concurrent_requests=1,
                max_input_megabytes=1,
                requests_per_second=1000,
            )
        )
        guard.validate_batch([object(), object()])
        with self.assertRaises(InputValidationError):
            guard.validate_batch([object(), object(), object()])
        with self.assertRaises(InputValidationError):
            guard.validate_input_bytes(2 * 1024 * 1024)

    def test_acquire_releases_semaphore(self):
        guard = ResourceGuard(
            ResourceLimits(max_batch_size=1, max_concurrent_requests=1, max_input_megabytes=1, requests_per_second=1000)
        )
        with guard.acquire():
            pass
        with guard.acquire():
            pass
