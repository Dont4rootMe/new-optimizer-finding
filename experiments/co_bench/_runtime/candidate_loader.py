"""Dynamic import + contract validation for CO-Bench organism candidates.

The CO-Bench candidate contract is ``def solve(**kwargs) -> dict``. CO-Bench
invokes the candidate as ``solve(**instance)`` (every instance key is passed by
keyword) and then ``eval_func(**instance, **solution)``. The pre-validation
here is a fast, clear-error gate before the (slower) subprocess evaluation.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


def import_module_from_path(path: str) -> ModuleType:
    """Import a Python module from an arbitrary file system path."""

    module_path = Path(path).expanduser().resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"Candidate file was not found: {module_path}")
    if module_path.suffix != ".py":
        raise ValueError(f"Candidate file must be a .py file, got: {module_path}")

    module_name = f"cobench_external_{hashlib.sha1(str(module_path).encode('utf-8')).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for candidate file: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_solve(path: str) -> tuple[Callable[..., Any], Path]:
    """Load and validate the CO-Bench ``solve(**kwargs) -> dict`` contract.

    Returns ``(solve, module_path)``.

    Signature rule (faithful to how CO-Bench calls ``solve(**instance)``):
      - ``solve`` must accept arbitrary keyword arguments (declare a
        ``**kwargs`` VAR_KEYWORD parameter), so any instance keys are tolerated.
      - Required positional-or-keyword / keyword-only parameters are allowed
        (e.g. ``def solve(nodes, **kwargs)``) because CO-Bench supplies every
        instance field by keyword; such a param is filled from the instance.
      - Required *positional-only* parameters are rejected: ``solve(**instance)``
        can never fill them by keyword.
    """

    module = import_module_from_path(path)
    module_path = Path(path).expanduser().resolve()

    solve = getattr(module, "solve", None)
    if solve is None or not callable(solve):
        raise AttributeError(
            f"Candidate module '{module_path}' must define a callable "
            "solve. Expected signature: def solve(**kwargs) -> dict."
        )

    try:
        signature = inspect.signature(solve)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Could not inspect solve in candidate module '{module_path}': {exc}"
        ) from exc

    params = list(signature.parameters.values())

    has_var_keyword = any(
        param.kind is inspect.Parameter.VAR_KEYWORD for param in params
    )
    if not has_var_keyword:
        raise TypeError(
            f"Candidate module '{module_path}' must define solve to accept "
            "arbitrary keyword arguments (a **kwargs parameter); CO-Bench calls "
            "solve(**instance). Expected signature: def solve(**kwargs) -> dict."
        )

    required_positional_only = [
        param.name
        for param in params
        if param.kind is inspect.Parameter.POSITIONAL_ONLY
        and param.default is inspect.Parameter.empty
    ]
    if required_positional_only:
        joined = ", ".join(required_positional_only)
        raise TypeError(
            f"Candidate module '{module_path}' defines solve with required "
            f"positional-only parameter(s): ({joined}). CO-Bench calls "
            "solve(**instance) by keyword, so positional-only params can never "
            "be filled. Expected signature: def solve(**kwargs) -> dict."
        )

    return solve, module_path
