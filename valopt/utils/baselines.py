"""Helpers for reading and wiring per-experiment baseline profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from valopt.utils.objective_tracking import safe_objective_float


def baseline_path(stats_root: str | Path, experiment_name: str) -> Path:
    """Return the canonical baseline path for one experiment."""

    return Path(stats_root).expanduser().resolve() / experiment_name / "baseline.json"


def validate_baseline_profile(payload: dict[str, Any], experiment_name: str) -> dict[str, Any]:
    """Validate the minimal fields needed for baseline-relative scoring."""

    objective_name = str(payload.get("objective_name", ""))
    if objective_name != "train_loss":
        raise ValueError(
            f"Baseline for '{experiment_name}' must use objective_name='train_loss'."
        )

    objective_direction = str(payload.get("objective_direction", ""))
    if objective_direction != "min":
        raise ValueError(
            f"Baseline for '{experiment_name}' must use objective_direction='min'."
        )

    objective_last = safe_objective_float(payload.get("objective_last"))
    if objective_last is None:
        raise ValueError(
            f"Baseline for '{experiment_name}' is missing a finite objective_last."
        )

    steps_raw = payload.get("steps")
    try:
        steps = int(steps_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Baseline for '{experiment_name}' is missing a valid integer steps value."
        ) from exc
    if steps <= 0:
        raise ValueError(
            f"Baseline for '{experiment_name}' must have steps > 0."
        )

    return payload


def load_baseline_profile(stats_root: str | Path, experiment_name: str) -> dict[str, Any]:
    """Load and validate the canonical baseline profile for one experiment."""

    path = baseline_path(stats_root, experiment_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing baseline profile for '{experiment_name}': {path}"
        )

    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_baseline_profile(payload, experiment_name)
    return payload


def inject_baseline_threshold(
    exp_cfg: DictConfig,
    *,
    stats_root: str | Path,
    experiment_name: str,
) -> None:
    """Attach baseline metadata to the experiment runtime config."""

    baseline_file = baseline_path(stats_root, experiment_name)
    exp_cfg.runtime.baseline_path = str(baseline_file)
    exp_cfg.runtime.baseline_last_train_loss = None
    exp_cfg.runtime.baseline_load_error = None

    try:
        payload = load_baseline_profile(stats_root, experiment_name)
    except (FileNotFoundError, ValueError) as exc:
        exp_cfg.runtime.baseline_load_error = str(exc)
        return

    exp_cfg.runtime.baseline_last_train_loss = float(payload["objective_last"])
