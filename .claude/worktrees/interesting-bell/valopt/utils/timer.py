"""Timing helper."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class WallTimer:
    """Simple wall-clock timer."""

    _start: float | None = None

    def start(self) -> None:
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        if self._start is None:
            return 0.0
        return time.perf_counter() - self._start
