"""Evolution subsystem for optimizer search."""

from __future__ import annotations

from typing import Any


def run_evolution(cfg: Any) -> dict:
    """Lazy import wrapper to avoid hydra dependency at package import time."""

    from .run import run_evolution as _run_evolution

    return _run_evolution(cfg)

__all__ = ["run_evolution"]
