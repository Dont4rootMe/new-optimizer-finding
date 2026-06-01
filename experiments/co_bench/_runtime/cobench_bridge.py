"""Lazy bridge to the (non-pip-installable) CO-Bench checkout.

CO-Bench (https://github.com/sunnweiwei/CO-Bench) ships top-level packages
``evaluation`` and ``agents`` and is *not* pip-installable. We therefore put
the checkout root on ``sys.path`` just long enough to import ``evaluation``
and cache the module. A missing checkout (or a missing solver dependency such
as ortools/networkx) is surfaced as an :class:`OptionalDependencyError` so the
host runner can mark the experiment ``skipped`` in smoke mode rather than
crashing the whole run.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional

import os

from experiments.co_bench._runtime.errors import OptionalDependencyError

# ``experiments/co_bench/_runtime/cobench_bridge.py`` -> parents[3] = project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Cache the imported ``evaluation`` module so repeat calls are cheap.
_EVALUATION_MODULE: Optional[ModuleType] = None


def _resolve_checkout_root() -> Path:
    """Resolve the CO-Bench checkout root (env override or vendored default)."""

    env_root = os.environ.get("COBENCH_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return (_PROJECT_ROOT / "third_party" / "CO-Bench").resolve()


def import_cobench() -> ModuleType:
    """Import and return CO-Bench's ``evaluation`` module.

    Raises :class:`OptionalDependencyError` (extra ``co_bench``) when the
    checkout is absent or a transitive solver dependency is missing.
    """

    global _EVALUATION_MODULE
    if _EVALUATION_MODULE is not None:
        return _EVALUATION_MODULE

    checkout_root = _resolve_checkout_root()
    if not checkout_root.is_dir() or not (checkout_root / "evaluation" / "__init__.py").is_file():
        raise OptionalDependencyError(
            "co_bench",
            "CO-Bench checkout not found at "
            f"'{checkout_root}'. Clone it and install the optional extra, e.g.:\n"
            "  git clone https://github.com/sunnweiwei/CO-Bench third_party/CO-Bench\n"
            '  pip install -e ".[co_bench]"\n'
            "or set COBENCH_ROOT to an existing checkout. The fastest path is "
            "scripts/bootstrap_cobench.sh.",
        )

    root_str = str(checkout_root)
    # Keep the checkout root on sys.path PERSISTENTLY (do not pop it). CO-Bench's
    # Evaluator spawns worker processes (ProcessPoolExecutor / mp.Process) that
    # re-import the ``evaluation`` package to unpickle the work items. With the
    # default ``spawn`` start method, children rebuild sys.path from the parent's
    # runtime ``sys.path`` (multiprocessing.spawn.get_preparation_data copies it),
    # so the entry must still be present when ``evaluate()`` runs. We also prepend
    # it to PYTHONPATH for robustness against any exec-based reconstruction.
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    _ensure_on_pythonpath(root_str)

    try:
        module = importlib.import_module("evaluation")
    except ImportError as exc:
        # CO-Bench's own imports (ortools, networkx, pulp, ...) are optional
        # solver deps we surface as the same optional-extra error.
        raise OptionalDependencyError(
            "co_bench",
            "Failed to import CO-Bench's 'evaluation' package: "
            f"{exc}. A solver dependency is likely missing. Install the "
            'optional extra ("pip install -e \\".[co_bench]\\"") or the deps '
            "listed in third_party/CO-Bench/requirements.txt.",
        ) from exc

    _EVALUATION_MODULE = module
    return module


def _ensure_on_pythonpath(root_str: str) -> None:
    """Prepend ``root_str`` to ``PYTHONPATH`` if not already present."""

    existing = os.environ.get("PYTHONPATH", "")
    parts = existing.split(os.pathsep) if existing else []
    if root_str in parts:
        return
    os.environ["PYTHONPATH"] = os.pathsep.join([root_str, *parts]) if parts else root_str
