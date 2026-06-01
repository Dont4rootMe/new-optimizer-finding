"""Dynamic import helpers for awtf2025 heuristic organism implementations."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
from pathlib import Path
from types import ModuleType
from typing import Callable


def import_module_from_path(path: str) -> ModuleType:
    """Import a Python module from an arbitrary file system path."""

    module_path = Path(path).expanduser().resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"Candidate file was not found: {module_path}")
    if module_path.suffix != ".py":
        raise ValueError(f"Candidate file must be a .py file, got: {module_path}")

    module_name = f"awtf2025_external_{hashlib.sha1(str(module_path).encode('utf-8')).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for candidate file: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_solve_case(path: str) -> tuple[Callable[[str], str], Path]:
    """Load strict awtf2025 `solve_case(input_text: str) -> str` contract."""

    module = import_module_from_path(path)
    module_path = Path(path).expanduser().resolve()

    solve_case = getattr(module, "solve_case", None)
    if solve_case is None or not callable(solve_case):
        raise AttributeError(
            f"Candidate module '{module_path}' must define callable solve_case(input_text: str) -> str."
        )

    try:
        signature = inspect.signature(solve_case)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Could not inspect solve_case in candidate module '{module_path}': {exc}"
        ) from exc

    parameters = list(signature.parameters.values())
    required = [
        param
        for param in parameters
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and param.default is inspect._empty
    ]
    keyword_required = [
        param for param in parameters if param.kind == inspect.Parameter.KEYWORD_ONLY and param.default is inspect._empty
    ]
    if len(required) != 1 or keyword_required:
        raise TypeError(
            f"Candidate module '{module_path}' must define solve_case(input_text: str) with exactly one "
            "required positional argument and no required keyword-only arguments."
        )

    return solve_case, module_path
