"""Structured JSON logging and operation records for Module B."""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_REDACT = {"authorization", "token", "secret", "password", "api_key"}


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: ("[REDACTED]" if k.lower() in _REDACT else _clean(v)) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def configure_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("human_activity_recognition")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger


def log_event(event: str, **fields: Any) -> None:
    payload = {"ts": time.time(), "event": event, **_clean(fields)}
    configure_logging().info(json.dumps(payload, ensure_ascii=True, sort_keys=True))


@contextmanager
def operation(action: str, **fields: Any) -> Iterator[dict[str, Any]]:
    started = time.perf_counter()
    record = {"action": action, "status": "started", **fields}
    log_event("operation.started", **record)
    try:
        yield record
    except Exception as exc:
        record.update(
            status="failed", error_type=type(exc).__name__, duration_ms=round((time.perf_counter() - started) * 1000, 3)
        )
        log_event("operation.failed", **record)
        raise
    else:
        record.update(status="succeeded", duration_ms=round((time.perf_counter() - started) * 1000, 3))
        log_event("operation.succeeded", **record)


def write_audit_record(path: Path, action: str, actor: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": time.time(), "action": action, "actor": actor, **_clean(fields)}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
