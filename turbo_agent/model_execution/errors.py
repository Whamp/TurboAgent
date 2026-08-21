"""Normalized model execution failures."""

from dataclasses import dataclass
from typing import Literal

from .types import ModelTarget

FailureKind = Literal[
    "authentication",
    "rate_limited",
    "timeout",
    "invalid_request",
    "unavailable",
    "malformed_response",
    "internal",
]


@dataclass(frozen=True)
class ModelExecutionError(Exception):
    """Safe, normalized failure from one model execution attempt.

    The message is safe for logs and request traces: adapters must not embed
    credentials, bearer tokens, or full provider payloads.
    """

    target: ModelTarget
    kind: FailureKind
    retryable: bool
    message: str
    retry_after_seconds: float | None = None

    def __str__(self) -> str:
        return (
            f"{self.kind} (target={self.target.name or self.target.id}): {self.message}"
        )
