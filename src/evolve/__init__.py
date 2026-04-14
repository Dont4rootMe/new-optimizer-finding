"""Evolution subsystem for optimizer search."""

from __future__ import annotations

from typing import Any


def run_evolution(cfg: Any) -> dict:
    """Lazy import wrapper to avoid hydra dependency at package import time."""

    from .run import run_evolution as _run_evolution

    return _run_evolution(cfg)


def run_seed_population(cfg: Any) -> dict:
    """Lazy import wrapper for seed-only population initialization."""

    from .seed_run import run_seed_population as _run_seed_population

    return _run_seed_population(cfg)


__all__ = ["run_evolution", "run_seed_population"]
