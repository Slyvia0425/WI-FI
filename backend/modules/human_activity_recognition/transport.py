"""Unified local/remote transport with timeout, retry, fallback and safe errors."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit

from .errors import TransportTimeoutError, TransportUnavailableError


class Transport(Protocol):
    def send(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...


@dataclass
class LocalTransport:
    handler: Callable[[Mapping[str, Any]], Mapping[str, Any]]

    def send(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return self.handler(payload)


@dataclass
class HttpTransport:
    endpoint: str
    token: str | None = None
    timeout_seconds: float = 8.0
    retries: int = 2
    backoff_seconds: float = 0.25
    opener: Any = urllib.request.urlopen

    def send(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        scheme = urlsplit(self.endpoint).scheme.lower()
        if scheme not in {"http", "https"}:
            raise TransportUnavailableError("remote endpoint must use http or https", {"scheme": scheme})
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(self.endpoint, data=data, headers=headers, method="POST")  # noqa: S310
            try:
                with self.opener(request, timeout=self.timeout_seconds) as response:
                    body = response.read().decode("utf-8")
                parsed = json.loads(body)
                if not isinstance(parsed, dict):
                    raise TransportUnavailableError("remote response is not a JSON object")
                return parsed
            except TimeoutError:
                last_error = TransportTimeoutError("remote request timed out", {"attempt": attempt + 1})
            except urllib.error.URLError as exc:
                if "timed out" in str(exc).lower():
                    last_error = TransportTimeoutError("remote request timed out", {"attempt": attempt + 1})
                else:
                    last_error = TransportUnavailableError(
                        "remote endpoint is unavailable", {"attempt": attempt + 1, "reason": str(exc)}
                    )
            except json.JSONDecodeError:
                last_error = TransportUnavailableError("remote response is not valid JSON", {"attempt": attempt + 1})
            if attempt < self.retries:
                time.sleep(self.backoff_seconds * (2**attempt))
        if isinstance(last_error, TransportTimeoutError):
            raise last_error
        raise last_error or TransportUnavailableError("remote request failed")


class ResilientTransport:
    """Try primary, then fallback only when policy permits it."""

    def __init__(self, primary: Transport, fallback: Transport | None = None, *, allow_fallback: bool = False) -> None:
        self.primary = primary
        self.fallback = fallback
        self.allow_fallback = allow_fallback

    def send(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            return self.primary.send(payload)
        except (TransportTimeoutError, TransportUnavailableError) as primary_error:
            if not self.allow_fallback or self.fallback is None:
                raise primary_error
            try:
                result = dict(self.fallback.send(payload))
                result.setdefault("degraded", True)
                result.setdefault("degradation_reason", primary_error.code)
                return result
            except Exception as fallback_error:
                raise TransportUnavailableError(
                    "primary and fallback transports failed",
                    {"primary": primary_error.code, "fallback": type(fallback_error).__name__},
                ) from fallback_error
