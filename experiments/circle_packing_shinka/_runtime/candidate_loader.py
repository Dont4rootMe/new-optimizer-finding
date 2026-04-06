"""Dynamic import helpers for circle-packing organism implementations."""

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

    module_name = f"circle_packing_external_{hashlib.sha1(str(module_path).encode('utf-8')).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for candidate file: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_run_packing(path: str) -> tuple[Callable[[], Any], Path]:
    """Load strict circle-packing `run_packing()` contract."""

    module = import_module_from_path(path)
    module_path = Path(path).expanduser().resolve()

    run_packing = getattr(module, "run_packing", None)
    if run_packing is None or not callable(run_packing):
        raise AttributeError(
            f"Candidate module '{module_path}' must define callable run_packing()."
        )

    try:
        signature = inspect.signature(run_packing)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Could not inspect run_packing in candidate module '{module_path}': {exc}"
        ) from exc

    required_positional = [
        param.name
        for param in signature.parameters.values()
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        and param.default is inspect._empty
    ]
    if required_positional:
        joined = ", ".join(required_positional)
        raise TypeError(
            f"Candidate module '{module_path}' must define run_packing() without required positional arguments; "
            f"got required params: ({joined})."
        )

    return run_packing, module_path
