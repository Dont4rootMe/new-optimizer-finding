"""Shared helpers for explicit root-config handling."""

from __future__ import annotations

from omegaconf import DictConfig


def ensure_root_runtime_config(cfg: DictConfig, *, context: str) -> None:
    """Require that a real top-level Hydra preset was loaded."""

    required_keys = {"paths", "experiments", "resources", "api_platforms", "evolver"}
    present_keys = set(cfg.keys()) if hasattr(cfg, "keys") else set()
    missing = sorted(required_keys.difference(present_keys))
    if missing:
        raise ValueError(
            f"{context} requires an explicit Hydra preset. "
            "Re-run with --config-name <preset> "
            "(for example: config_optimization_survey, config_circle_packing_shinka)."
        )
