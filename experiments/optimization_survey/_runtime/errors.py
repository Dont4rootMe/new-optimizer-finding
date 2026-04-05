"""Experiment-scoped error types."""

from __future__ import annotations


class OptionalDependencyError(RuntimeError):
    """Raised when an optional extra is required for an experiment."""

    def __init__(self, extra_name: str, message: str) -> None:
        super().__init__(message)
        self.extra_name = extra_name

