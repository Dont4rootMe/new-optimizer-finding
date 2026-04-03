"""Tests for dynamic optimizer import."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from optbench.utils.import_utils import load_optimizer_builder

ROOT = Path(__file__).resolve().parents[1]


def test_import_optimizer_factory() -> None:
    optimizer_path = ROOT / "optimizer_guesses" / "examples" / "sgd_baseline.py"
    builder, resolved_path, optimizer_name = load_optimizer_builder(str(optimizer_path))

    assert callable(builder)

    model = nn.Linear(4, 2)
    controller = builder(model, 100)

    assert callable(getattr(controller, "step", None))
    assert callable(getattr(controller, "zero_grad", None))

    assert resolved_path == optimizer_path.resolve()
    assert optimizer_name == "SGDBaselineController"


def test_import_optimizer_rejects_legacy_builder_contract(tmp_path: Path) -> None:
    optimizer_path = tmp_path / "legacy_optimizer.py"
    optimizer_path.write_text(
        (
            "class LegacyController:\n"
            "    def __init__(self, cfg):\n"
            "        self.cfg = cfg\n\n"
            "    def step(self, weights, grads, activations, step_fn):\n"
            "        del weights, grads, activations, step_fn\n\n"
            "    def zero_grad(self, set_to_none=True):\n"
            "        del set_to_none\n\n"
            "def build_optimizer(cfg):\n"
            "    return LegacyController(cfg)\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="build_optimizer"):
        load_optimizer_builder(str(optimizer_path))


def test_import_optimizer_rejects_missing_step_fn(tmp_path: Path) -> None:
    optimizer_path = tmp_path / "missing_step_fn.py"
    optimizer_path.write_text(
        (
            "class MissingStepFnController:\n"
            "    def __init__(self, model, max_steps):\n"
            "        self.model = model\n"
            "        self.max_steps = max_steps\n\n"
            "    def step(self, weights, grads, activations):\n"
            "        del weights, grads, activations\n\n"
            "    def zero_grad(self, set_to_none=True):\n"
            "        del set_to_none\n\n"
            "def build_optimizer(model, max_steps):\n"
            "    return MissingStepFnController(model, max_steps)\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="controller.step"):
        builder, _, _ = load_optimizer_builder(str(optimizer_path))
        builder(nn.Linear(4, 2), 100)


def test_import_utils_no_outdated_builder_alias() -> None:
    import optbench.utils.import_utils as import_utils

    assert not hasattr(import_utils, "BuildOptimizerCallable")
