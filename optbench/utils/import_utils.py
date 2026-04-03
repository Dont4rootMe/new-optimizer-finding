"""Dynamic import helpers for optimizer-controller modules."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import torch

from optbench.optimizer_api import OptimizerBuilder, OptimizerControllerProtocol


def _validate_signature(
    fn: Callable[..., Any],
    *,
    expected_params: list[str],
    label: str,
    module_path: Path,
    contract_hint: str | None = None,
) -> None:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Could not inspect {label} in optimizer module '{module_path}': {exc}"
        ) from exc

    positional = [
        param
        for param in signature.parameters.values()
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    actual = [param.name for param in positional]
    if actual != expected_params:
        suffix = f" {contract_hint}" if contract_hint else ""
        raise TypeError(
            f"Optimizer module '{module_path}' must define {label}"
            f" with signature ({', '.join(expected_params)}); got ({', '.join(actual)})."
            f"{suffix}"
        )


def import_module_from_path(path: str) -> ModuleType:
    """Import a Python module from an arbitrary file system path."""

    module_path = Path(path).expanduser().resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"Optimizer file was not found: {module_path}")
    if module_path.suffix != ".py":
        raise ValueError(f"Optimizer file must be a .py file, got: {module_path}")

    module_name = f"optbench_external_{hashlib.sha1(str(module_path).encode('utf-8')).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for optimizer file: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_controller(controller: Any, module_path: Path) -> None:
    _validate_signature(
        controller.step,
        expected_params=["weights", "grads", "activations", "step_fn"],
        label="controller.step",
        module_path=module_path,
        contract_hint="step(weights, grads, activations, step_fn) is required.",
    )
    _validate_signature(
        controller.zero_grad,
        expected_params=["set_to_none"],
        label="controller.zero_grad",
        module_path=module_path,
    )

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
    _validate_signature(
        build_optimizer,
        expected_params=["model", "max_steps"],
        label="build_optimizer",
        module_path=module_path,
        contract_hint="build_optimizer(model, max_steps) is required.",
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
