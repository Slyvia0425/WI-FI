"""Bounded concurrency, rate limiting and input-size checks."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

from .errors import InputValidationError


@dataclass(frozen=True)
class ResourceLimits:
    max_batch_size: int
    max_concurrent_requests: int
    max_input_megabytes: int
    requests_per_second: float


class RateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        if requests_per_second <= 0:
            raise InputValidationError("requests_per_second must be positive")
        self.interval = 1.0 / requests_per_second
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self.interval - (now - self._last)
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()


class ResourceGuard:
    def __init__(self, limits: ResourceLimits) -> None:
        self.limits = limits
        self._semaphore = threading.BoundedSemaphore(max(1, limits.max_concurrent_requests))
        self._limiter = RateLimiter(limits.requests_per_second)

    @contextmanager
    def acquire(self) -> Iterator[None]:
        self._limiter.wait()
        with self._semaphore:
            yield

    def validate_batch(self, windows: Sequence[object]) -> None:
        if not windows:
            raise InputValidationError("batch must not be empty")
        if len(windows) > self.limits.max_batch_size:
            raise InputValidationError(
                "batch exceeds configured maximum", {"size": len(windows), "maximum": self.limits.max_batch_size}
            )

    def validate_input_bytes(self, size: int) -> None:
        maximum = self.limits.max_input_megabytes * 1024 * 1024
        if size > maximum:
            raise InputValidationError("input exceeds configured memory limit", {"bytes": size, "maximum": maximum})
