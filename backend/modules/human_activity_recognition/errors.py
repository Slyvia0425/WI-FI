"""Typed, classified errors for Module B. No caller should need to parse a traceback."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModuleBError(Exception):
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"

    def to_payload(self) -> dict[str, Any]:
        return {
            "ok": False,
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            },
        }


class ConfigurationError(ModuleBError):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__("CFG_INVALID", message, details or {})


class InputValidationError(ModuleBError):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__("INPUT_INVALID", message, details or {})


class DataNotFoundError(ModuleBError):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__("DATA_NOT_FOUND", message, details or {})


class DataValidationError(ModuleBError):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__("DATA_INVALID", message, details or {})


class SplitLeakageError(ModuleBError):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__("SPLIT_LEAKAGE", message, details or {})


class ModelUnavailableError(ModuleBError):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__("MODEL_UNAVAILABLE", message, details or {})


class ModelValidationError(ModuleBError):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__("MODEL_INVALID", message, details or {})


class InferenceError(ModuleBError):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__("INFERENCE_FAILED", message, details or {})


class TransportTimeoutError(ModuleBError):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__("TRANSPORT_TIMEOUT", message, details or {})


class TransportUnavailableError(ModuleBError):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__("TRANSPORT_UNAVAILABLE", message, details or {})


class AuthorizationError(ModuleBError):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__("AUTH_DENIED", message, details or {})


class ExportValidationError(ModuleBError):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__("EXPORT_INVALID", message, details or {})
