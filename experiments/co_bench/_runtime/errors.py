"""Experiment-scoped error types for the CO-Bench family.

Re-export ``OptionalDependencyError`` so the host runner's smoke-skip path
(``src/validate/runner.py`` matches ``exc.__class__.__name__ ==
"OptionalDependencyError"``) sees the same class regardless of which
experiment family raised it.
"""

from __future__ import annotations

from experiments.optimization_survey._runtime.errors import OptionalDependencyError

__all__ = ["OptionalDependencyError"]
