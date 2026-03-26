"""Dynamic import helpers for optimizer-controller modules."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import torch

from valopt.optimizer_api import OptimizerBuilder, OptimizerControllerProtocol

BuildOptimizerCallable = Callable[[Any], OptimizerControllerProtocol]


def import_module_from_path(path: str) -> ModuleType:
    """Import a Python module from an arbitrary file system path."""

    module_path = Path(path).expanduser().resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"Optimizer file was not found: {module_path}")
    if module_path.suffix != ".py":
        raise ValueError(f"Optimizer file must be a .py file, got: {module_path}")

    module_name = f"valopt_external_{hashlib.sha1(str(module_path).encode('utf-8')).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for optimizer file: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_controller(controller: Any, module_path: Path) -> None:
    missing = [
        method
        for method in ("step", "zero_grad")
        if not callable(getattr(controller, method, None))
    ]
    if missing:
        missing_str = ", ".join(missing)
        raise AttributeError(
            f"Optimizer module '{module_path}' returned invalid controller. "
            f"Missing callable methods: {missing_str}."
        )


def load_optimizer_builder(path: str) -> tuple[OptimizerBuilder, Path, str]:
    """Load strict `build_optimizer(model, max_steps) -> controller` contract from Python file."""

    module = import_module_from_path(path)
    module_path = Path(path).expanduser().resolve()

    build_optimizer = getattr(module, "build_optimizer", None)
    if build_optimizer is None or not callable(build_optimizer):
        raise AttributeError(
            f"Optimizer module '{module_path}' must define callable build_optimizer(model, max_steps)."
        )

    optimizer_name_obj = getattr(module, "OPTIMIZER_NAME", module_path.stem)
    optimizer_name = str(optimizer_name_obj)

    def builder(model: torch.nn.Module, max_steps: int) -> OptimizerControllerProtocol:
        controller = build_optimizer(model, max_steps)
        _validate_controller(controller, module_path)
        return controller

    return builder, module_path, optimizer_name


def load_optimizer_factory(path: str) -> tuple[OptimizerBuilder, Path, str]:
    """Backward-compatible alias for older call sites."""

    return load_optimizer_builder(path)
