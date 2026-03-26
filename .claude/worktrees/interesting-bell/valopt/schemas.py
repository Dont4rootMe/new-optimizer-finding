"""Shared typed schemas and validation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


class OptionalDependencyError(RuntimeError):
    """Raised when an optional extra is required for an experiment."""

    def __init__(self, extra_name: str, message: str) -> None:
        super().__init__(message)
        self.extra_name = extra_name


@dataclass(slots=True)
class RunResult:
    """Canonical run result persisted to JSON."""

    run_id: str
    timestamp: str
    experiment_name: str
    optimizer_path: str
    optimizer_name: str
    final_quality: float
    target_quality: float
    converged: bool
    steps: int
    wall_time_sec: float
    best_quality: float
    seed: int
    device: str
    precision: str
    resolved_config_path: str
    extra_metrics: dict[str, Any]
    safety_flags: dict[str, Any]
    final_metrics: dict[str, Any] = field(default_factory=dict)
    best_metrics: dict[str, Any] = field(default_factory=dict)
    samples_or_tokens_seen: int = 0
    steps_to_target: int | None = None
    status: str = "ok"
    error_msg: str | None = None
    smoke: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert dataclass to JSON-serializable dict."""
        return asdict(self)


_REQUIRED_FIELDS: tuple[tuple[str, type | tuple[type, ...]], ...] = (
    ("run_id", str),
    ("timestamp", str),
    ("experiment_name", str),
    ("optimizer_path", str),
    ("optimizer_name", str),
    ("final_quality", (int, float)),
    ("target_quality", (int, float)),
    ("converged", bool),
    ("steps", int),
    ("wall_time_sec", (int, float)),
    ("best_quality", (int, float)),
    ("seed", int),
    ("device", str),
    ("precision", str),
    ("resolved_config_path", str),
    ("extra_metrics", dict),
    ("safety_flags", dict),
)


def validate_run_result_dict(data: dict[str, Any]) -> None:
    """Validate required keys and basic field types for a run result payload."""

    for field_name, expected_type in _REQUIRED_FIELDS:
        if field_name not in data:
            raise ValueError(f"Run result is missing required field '{field_name}'.")
        if not isinstance(data[field_name], expected_type):
            expected_name = (
                ", ".join(t.__name__ for t in expected_type)
                if isinstance(expected_type, tuple)
                else expected_type.__name__
            )
            actual_name = type(data[field_name]).__name__
            raise ValueError(
                f"Run result field '{field_name}' expected {expected_name}, got {actual_name}."
            )

    status = data.get("status", "ok")
    if status not in {"ok", "skipped", "failed"}:
        raise ValueError("Run result field 'status' must be one of: ok, skipped, failed.")

    if "smoke" in data and not isinstance(data["smoke"], bool):
        raise ValueError("Run result field 'smoke' must be a bool when provided.")

    if "steps_to_target" in data and data["steps_to_target"] is not None and not isinstance(
        data["steps_to_target"], int
    ):
        raise ValueError("Run result field 'steps_to_target' must be int or null.")
